import pprint
import argparse
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.agent import ServerAgent
from appfl.comm.globus_compute import GlobusComputeServerCommunicator
import torch
import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config",
    type=str,
    default="./resources/config_gc/llm_chem/server_fedavg_full.yaml",
)
argparser.add_argument(
    "--client_config", type=str, default="./resources/config_gc/llm_chem/clients_fedavg_full.yaml"
)
argparser.add_argument("--compute_token", required=False)
argparser.add_argument("--openid_token", required=False)
args = argparser.parse_args()

# Load server and client agents configurations
server_agent_config = OmegaConf.load(args.server_config)
client_agent_configs = OmegaConf.load(args.client_config)

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server communicator
server_communicator = GlobusComputeServerCommunicator(
    server_agent_config=server_agent.server_agent_config,
    client_agent_configs=client_agent_configs["clients"],
    logger=server_agent.logger,
    **(
        {
            "compute_token": args.compute_token,
            "openid_token": args.openid_token,
        }
        if args.compute_token is not None and args.openid_token is not None
        else {}
    ),
)

# Get sample size from clients
# server_communicator.send_task_to_all_clients(task_name="get_sample_size_ds")
# sample_size_ret = server_communicator.recv_result_from_all_clients()[1]
# for client_endpoint_id, sample_size in sample_size_ret.items():
#     server_agent.set_sample_size(client_endpoint_id, sample_size["sample_size"])

server_agent.set_sample_size("76aadeb8-2852-452e-ae73-a9408f3cfbd9", 115200) 
server_agent.set_sample_size("fedab150-fb9f-43c9-9ea9-038b17be1348", 224256)
server_agent.set_sample_size("7af621d6-d761-450c-b955-0c9fed189cd8", 114688) 
server_agent.set_sample_size("b303b57e-d3c2-4e5f-9ea0-df9ce2a12654", 1505280) 

metadata = {"round": 0}

# Train the model
server_communicator.send_task_to_all_clients(
    task_name="train_ds",
    model=server_agent.get_parameters(globus_compute_run=True),
    metadata=metadata,
    need_model_response=True,
)

client_id_name_map = {"76aadeb8-2852-452e-ae73-a9408f3cfbd9": "Polaris", 
                      "fedab150-fb9f-43c9-9ea9-038b17be1348": "Perlmutter",
                      "7af621d6-d761-450c-b955-0c9fed189cd8": "Frontier", 
                      "b303b57e-d3c2-4e5f-9ea0-df9ce2a12654": "Aurora"
                      }

model_futures = {}
client_rounds = {}
while not server_agent.training_finished():
    client_endpoint_id, client_model, client_metadata = (
        server_communicator.recv_result_from_one_client()
    )
    server_agent.logger.info(
        f"Received model from client {client_endpoint_id}, with metadata:\n{pprint.pformat(client_metadata)}"
    )
    global_model = server_agent.global_update(
        client_endpoint_id,
        client_model,
        **client_metadata,
    )
    if isinstance(global_model, Future):
        model_futures[client_endpoint_id] = global_model
    else:
        if isinstance(global_model, tuple):
            global_model, metadata = global_model
        else:
            metadata = {}
        if client_endpoint_id not in client_rounds:
            client_rounds[client_endpoint_id] = 0
        client_rounds[client_endpoint_id] += 1
        metadata["round"] = client_rounds[client_endpoint_id]

        client_name = client_id_name_map.get(client_endpoint_id, client_endpoint_id)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        torch.save(global_model, f"") # Save the global model to the path specified

        if not server_agent.training_finished():
            server_communicator.send_task_to_one_client(
                client_endpoint_id,
                task_name="train_ds",
                model=global_model,
                metadata=metadata,
                need_model_response=True,
            )
    # Deal with the model futures
    del_keys = []
    for client_endpoint_id in model_futures:
        if model_futures[client_endpoint_id].done():
            global_model = model_futures[client_endpoint_id].result()
            if isinstance(global_model, tuple):
                global_model, metadata = global_model
            else:
                metadata = {}
            if client_endpoint_id not in client_rounds:
                client_rounds[client_endpoint_id] = 0
            client_rounds[client_endpoint_id] += 1
            metadata["round"] = client_rounds[client_endpoint_id]

            client_name = client_id_name_map.get(client_endpoint_id, client_endpoint_id)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            torch.save(global_model, f"") # Save the global model to the path sepcified

            if not server_agent.training_finished():
                server_communicator.send_task_to_one_client(
                    client_endpoint_id,
                    task_name="train_ds",
                    model=global_model,
                    metadata=metadata,
                    need_model_response=True,
                )
            del_keys.append(client_endpoint_id)
    for key in del_keys:
        model_futures.pop(key)

server_communicator.cancel_all_tasks()
server_communicator.shutdown_all_clients()
