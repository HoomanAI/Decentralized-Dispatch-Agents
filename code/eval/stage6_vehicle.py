"""Vehicle-centric Stage 6 gates and training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode
from code.env.vehicle_centric import ProtocolMyopic, run_vehicle_episode
from code.eval.stage6 import _perturb_exposure
from code.policy.vehicle_centric import NeuralVehicleController, ResidualVehiclePolicy, VehicleActorQCritic, vehicle_state


class TeacherVehicleController:
    def __init__(self, model: VehicleActorQCritic, edge_count: int) -> None:
        self.model, self.edge_count = model, edge_count
        self.samples: list[tuple[dict[str, torch.Tensor], int]] = []

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        allowed = context["allowed_indices"]
        action = max(allowed, key=lambda i: context["candidates"][i].immediate_survival)
        self.samples.append((vehicle_state(context, self.edge_count), action))
        return action, float(context["candidates"][action].immediate_survival)

    def observe(self, reward: float, terminal: bool = False) -> None:
        del reward, terminal


def _setup(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path):
    pc, nc, dc = config["policy_training"], config["reduced_network"], config["reduced_demand"]
    graph, hospitals, lookup = build_runtime_network(data_dir / "reduced/synthetic_road_features.geojson", data_dir / "reduced/synthetic_fire_fronts.geojson", nc)
    pds = affected.query("region == 'PDS'").set_index("day")
    union = float(pds["Total"].iloc[0])
    age = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    env = dict(config["environment"])
    env.update({"scenario_alpha": float(pc["scenario_alpha"]), "scenario_gamma": float(pc["scenario_gamma"]), "phi_min": float(pc["phi_min"]), "scenario_alpha_concentration": pc["scenario_alpha_concentration"], "operational_minutes": float(dc["operational_minutes"]), "fleet_by_class": dict(pc["fleet_by_class"])})
    nodes = np.array([n for n, a in graph.nodes(data=True) if not a.get("hospital")])
    return pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes


def run_vehicle_gates(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path, clone_episodes: int = 500, reuse_clone: bool = False):
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(affected, config, data_dir)
    fleet_count = sum(int(v) for v in pc["fleet_by_class"].values())
    direct_rows = []
    for seed in range(int(pc["evaluation_seeds"])):
        direct_total = pipeline_total = centralized_total = 0.0
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
            patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc, patient_seed)
            direct, _ = run_vehicle_episode(graph, hospitals, lookup, patients, day, [ProtocolMyopic() for _ in range(fleet_count)], nc, config["belief"], env, float(pds.loc[day, "affected_population"]), seed=patient_seed)
            pipeline, _ = run_vehicle_episode(graph, hospitals, lookup, patients, day, [ProtocolMyopic() for _ in range(fleet_count)], nc, config["belief"], env, float(pds.loc[day, "affected_population"]), seed=patient_seed)
            centralized, _ = run_episode(graph, hospitals, lookup, patients, day, "belief_aware", nc, config["belief"], env, float(pds.loc[day, "affected_population"]))
            direct_total += direct["survival_total"]; pipeline_total += pipeline["survival_total"]
            centralized_total += centralized["survival_total"]
        direct_rows.append({"seed": seed, "direct_protocol_myopic": direct_total, "pipeline_protocol_myopic": pipeline_total, "centralized_myopic_same_instance": centralized_total, "difference": pipeline_total-direct_total, "protocol_minus_centralized": direct_total-centralized_total})
    interface = pd.DataFrame(direct_rows)
    interface.to_csv(results_dir / "stage6_vehicle_interface_gate.csv", index=False)
    assert float(interface.difference.abs().max()) == 0.0

    torch.manual_seed(int(pc["seed"]) + 400000)
    edge_count = graph.number_of_edges()
    model = VehicleActorQCritic(edge_count, int(pc["hidden_dim"]), fleet_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    modes = list(pc["augmentation_modes"])
    augmented = {mode: _perturb_exposure(graph, data_dir / "reduced/synthetic_fire_fronts.geojson", mode, float(pc["augmentation_buffer_degrees"]), float(pc["augmentation_translation_degrees"])) for mode in set(modes)}
    logs = []
    checkpoint = results_dir / "stage6_vehicle_clone.pt"
    if reuse_clone and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
    for episode in range(0 if reuse_clone and checkpoint.exists() else clone_episodes):
        day = int(pc["train_days"][episode % len(pc["train_days"])])
        mode = modes[episode % len(modes)]; train_graph = augmented[mode]
        train_lookup = {int(a["edge_id"]): i for i, (_, _, a) in enumerate(train_graph.edges(data=True))}
        patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc, int(pc["seed"]) + 400000 + episode)
        teachers = [TeacherVehicleController(model, edge_count) for _ in range(fleet_count)]
        run_vehicle_episode(train_graph, hospitals, train_lookup, patients, day, teachers, nc, config["belief"], env, float(pds.loc[day, "affected_population"]), seed=episode)
        samples = [sample for teacher in teachers for sample in teacher.samples]
        if not samples: continue
        losses=[]; correct=0; ent=[]
        for state, label in samples:
            logits, _ = model(state); losses.append(functional.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))); correct += int(torch.argmax(logits).item()==label); ent.append(float(torch.distributions.Categorical(logits=logits).entropy().item()/max(np.log(len(logits)),1.0)))
        loss=torch.stack(losses).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        logs.append({"episode":episode+1,"examples":len(samples),"loss":float(loss.item()),"agreement":correct/len(samples),"normalized_entropy":float(np.mean(ent))})
    if logs:
        pd.DataFrame(logs).to_csv(results_dir / "stage6_vehicle_clone_training.csv", index=False)
    eval_rows=[]
    model.eval()
    for seed in range(int(pc["evaluation_seeds"])):
        protocol_total=clone_total=0.0; agreements=[]; entropies=[]
        for day in range(8,13):
            patient_seed=int(pc["seed"])+300000+seed*100+day
            patients=generate_patients(day,float(pds.loc[day,"affected_population"]),union,nodes,age,dc,patient_seed)
            protocol,_=run_vehicle_episode(graph,hospitals,lookup,patients,day,[ProtocolMyopic() for _ in range(fleet_count)],nc,config["belief"],env,float(pds.loc[day,"affected_population"]),seed=patient_seed)
            controls=[NeuralVehicleController(model,edge_count,greedy=True) for _ in range(fleet_count)]
            cloned,_=run_vehicle_episode(graph,hospitals,lookup,patients,day,controls,nc,config["belief"],env,float(pds.loc[day,"affected_population"]),seed=patient_seed)
            protocol_total+=protocol["survival_total"]; clone_total+=cloned["survival_total"]
            for control in controls:
                entropies.extend(control.normalized_entropies)
                agreements.extend(control.myopic_agreements)
        eval_rows.append({"seed":seed,"protocol_myopic":protocol_total,"clone":clone_total,"difference":clone_total-protocol_total,"agreement":float(np.mean(agreements)),"normalized_entropy":float(np.mean(entropies))})
    evaluation=pd.DataFrame(eval_rows); evaluation.to_csv(results_dir / "stage6_vehicle_clone_evaluation.csv",index=False)
    torch.save({"model_state_dict":model.state_dict(),"edge_count":edge_count,"fleet_count":fleet_count,"clone_episodes":clone_episodes,"synthetic_input":True},results_dir/"stage6_vehicle_clone.pt")
    return results_dir/"stage6_vehicle_interface_gate.csv",results_dir/"stage6_vehicle_clone_evaluation.csv"


def _vehicle_update(model, optimizer, transitions, mode, clip_ratio, value_weight, kl_weight, minibatch_size):
    selected = np.random.choice(len(transitions), size=min(minibatch_size, len(transitions)), replace=False)
    losses=[]; actors=[]; critics=[]; kls=[]; entropies=[]
    for index in selected:
        item=transitions[int(index)]; state=item["state"]
        logits,q=model(state)
        allowed_mask=state.get("allowed_mask",torch.ones_like(logits,dtype=torch.bool))
        masked_logits=torch.where(allowed_mask,logits,torch.full_like(logits,-1.0e9))
        temperature=state.get("behavior_temperature",torch.tensor(1.0))
        distribution=torch.distributions.Categorical(logits=masked_logits/temperature)
        action=torch.tensor(item["action"]); ratio=torch.exp(distribution.log_prob(action)-item["old_logp"])
        target=torch.tensor(item["target"],dtype=torch.float32)
        if mode.startswith("counterfactual"):
            advantage=(q[action]-torch.sum(distribution.probs*q)).detach()
        else:
            advantage=(target-q.mean()).detach()
        actor=-torch.minimum(ratio*advantage,torch.clamp(ratio,1-clip_ratio,1+clip_ratio)*advantage)
        critic=(q[action]-target)**2
        with torch.no_grad():
            base_logits,_=model.clone(state)
            base_masked=torch.where(allowed_mask,base_logits,torch.full_like(base_logits,-1.0e9))
            base_distribution=torch.distributions.Categorical(logits=base_masked/temperature)
        kl=torch.distributions.kl_divergence(distribution,base_distribution)
        losses.append(actor+value_weight*critic+kl_weight*kl); actors.append(actor); critics.append(critic); kls.append(kl); entropies.append(distribution.entropy())
    loss=torch.stack(losses).mean(); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return {"loss":float(loss.item()),"actor_loss":float(torch.stack(actors).mean().item()),"critic_loss":float(torch.stack(critics).mean().item()),"kl":float(torch.stack(kls).mean().item()),"entropy":float(torch.stack(entropies).mean().item())}


def train_vehicle_residual(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path, mode: str, updates: int, kl_override: float | None = None):
    pc,nc,dc,graph,hospitals,lookup,pds,union,age,env,nodes=_setup(affected,config,data_dir)
    fleet_count=sum(int(v) for v in pc["fleet_by_class"].values()); edge_count=graph.number_of_edges()
    saved=torch.load(results_dir/"stage6_vehicle_clone.pt",map_location="cpu",weights_only=False)
    clone=VehicleActorQCritic(edge_count,int(pc["hidden_dim"]),fleet_count); clone.load_state_dict(saved["model_state_dict"])
    model=ResidualVehiclePolicy(clone); optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=float(pc["learning_rate"]))
    modes=list(pc["augmentation_modes"]); augmented={m:_perturb_exposure(graph,data_dir/"reduced/synthetic_fire_fronts.geojson",m,float(pc["augmentation_buffer_degrees"]),float(pc["augmentation_translation_degrees"])) for m in set(modes)}
    buffer=[]; logs=[]; episode=0; steps=0; recent_survival=[]; recent_normalized_entropy=[]
    entropy_control = mode == "counterfactual_softened"
    sampling_temperature = 1.0
    while len(logs)<updates:
        day=int(pc["train_days"][episode%len(pc["train_days"])]); aug=modes[episode%len(modes)]; g=augmented[aug]
        gl={int(a["edge_id"]):i for i,(_,_,a) in enumerate(g.edges(data=True))}; patient_seed=int(pc["seed"])+500000+episode
        patients=generate_patients(day,float(pds.loc[day,"affected_population"]),union,nodes,age,dc,patient_seed)
        controls=[NeuralVehicleController(model,edge_count,greedy=False,temperature=sampling_temperature) for _ in range(fleet_count)]
        episode_summary,_=run_vehicle_episode(g,hospitals,gl,patients,day,controls,nc,config["belief"],env,float(pds.loc[day,"affected_population"]),seed=patient_seed)
        recent_survival.append(float(episode_summary["survival_total"]))
        recent_normalized_entropy.extend(value for control in controls for value in control.normalized_entropies)
        for control in controls:
            sequence=control.transitions; next_target=0.0
            for item in reversed(sequence):
                selected_q=float(item["old_q"][item["action"]])
                if item["terminal"]: next_target=0.0
                target=item["reward"]+float(pc["gae_lambda"])*next_target+(1-float(pc["gae_lambda"]))*selected_q
                item["target"]=target; next_target=target
            buffer.extend(sequence)
        steps+=sum(len(c.transitions) for c in controls); episode+=1
        if len(buffer)>=int(pc["steps_per_update"]):
            progress=len(logs)/max(updates-1,1); kl_weight=float(kl_override) if kl_override is not None else 0.05*(1-progress)
            metrics=_vehicle_update(model,optimizer,buffer,mode,float(pc["clip_ratio"]),float(pc["value_weight"]),kl_weight,int(pc["minibatch_size"]))
            observed_entropy=float(np.mean(recent_normalized_entropy))
            target_entropy=np.nan
            if entropy_control:
                if progress <= 2.0/3.0:
                    target_entropy=0.18
                else:
                    fraction=(progress-2.0/3.0)/(1.0/3.0)
                    target_entropy=0.18+(0.09-0.18)*min(max(fraction,0.0),1.0)
                sampling_temperature=float(np.clip(sampling_temperature*np.exp(2.0*(target_entropy-observed_entropy)),0.25,100.0))
            logs.append({"update":len(logs)+1,"environment_steps":steps,"kl_weight":kl_weight,"sampling_temperature":sampling_temperature,"target_normalized_entropy":target_entropy,"mean_episode_survival":float(np.mean(recent_survival)),"mean_normalized_entropy":observed_entropy,**metrics}); buffer=[]; recent_survival=[]; recent_normalized_entropy=[]
    path=results_dir/f"stage6_vehicle_{mode}_training.csv"; pd.DataFrame(logs).to_csv(path,index=False)
    torch.save({"model_state_dict":model.state_dict(),"mode":mode,"updates":updates,"environment_steps":steps,"synthetic_input":True},results_dir/f"stage6_vehicle_{mode}.pt")
    return path


def evaluate_vehicle_residual(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path, mode: str):
    """Evaluate one saved residual arm against protocol myopic on paired seeds."""
    pc,nc,dc,graph,hospitals,lookup,pds,union,age,env,nodes=_setup(affected,config,data_dir)
    fleet_count=sum(int(v) for v in pc["fleet_by_class"].values()); edge_count=graph.number_of_edges()
    clone_saved=torch.load(results_dir/"stage6_vehicle_clone.pt",map_location="cpu",weights_only=False)
    clone=VehicleActorQCritic(edge_count,int(pc["hidden_dim"]),fleet_count); clone.load_state_dict(clone_saved["model_state_dict"])
    model=ResidualVehiclePolicy(clone)
    saved=torch.load(results_dir/f"stage6_vehicle_{mode}.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    rows=[]
    for seed in range(int(pc["evaluation_seeds"])):
        protocol_total=policy_total=0.0; agreements=[]; entropies=[]; duplicates=[]; divergences=[]; clone_kls=[]; clone_agreements=[]
        for day in range(8,13):
            patient_seed=int(pc["seed"])+300000+seed*100+day
            patients=generate_patients(day,float(pds.loc[day,"affected_population"]),union,nodes,age,dc,patient_seed)
            protocol,_=run_vehicle_episode(graph,hospitals,lookup,patients,day,[ProtocolMyopic() for _ in range(fleet_count)],nc,config["belief"],env,float(pds.loc[day,"affected_population"]),seed=patient_seed)
            controls=[NeuralVehicleController(model,edge_count,greedy=True,reference_model=clone) for _ in range(fleet_count)]
            policy,_=run_vehicle_episode(graph,hospitals,lookup,patients,day,controls,nc,config["belief"],env,float(pds.loc[day,"affected_population"]),seed=patient_seed)
            protocol_total+=protocol["survival_total"]; policy_total+=policy["survival_total"]
            duplicates.append(policy["duplicate_assignment_rate"]); divergences.append(policy["mean_belief_divergence_at_reconnection"])
            for control in controls:
                agreements.extend(control.myopic_agreements); entropies.extend(control.normalized_entropies); clone_kls.extend(control.reference_kls); clone_agreements.extend(control.reference_argmax_agreements)
        rows.append({"synthetic_input":True,"seed":seed,"protocol_myopic":protocol_total,"policy":policy_total,"paired_difference":policy_total-protocol_total,"argmax_agreement":float(np.mean(agreements)),"normalized_entropy":float(np.mean(entropies)),"mean_kl_to_clone":float(np.mean(clone_kls)),"argmax_agreement_with_clone":float(np.mean(clone_agreements)),"duplicate_assignment_rate":float(np.mean(duplicates)),"belief_divergence":float(np.mean(divergences))})
    path=results_dir/f"stage6_vehicle_{mode}_evaluation.csv"; pd.DataFrame(rows).to_csv(path,index=False); return path
