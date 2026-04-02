import json
import urllib.request
import random

# ComfyUI local service address
SERVER_ADDRESS = "127.0.0.1:8000"


def send_prompt(prompt_workflow):
    """Pack the revised blueprint and send it to ComfyUI"""
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def main():
    # 1. Load blueprint
    print("Loading workflow_api.json...")
    with open("PCB_1.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # 2. Dynamically modify parameters
    new_seed = random.randint(1, 9999999999)  # 随机生成一个新种子

    sampler_node_id = "3"

    try:
        workflow[sampler_node_id]["inputs"]["seed"] = new_seed
    except KeyError:
        print(f"Warning: The node with ID '{sampler_node_id}' could not be found, or the node does not have a seed parameter! Please check the JSON.")
        return

    # 3. Send request
    print("Sending composition instructions to ComfyUI...")
    try:
        response = send_prompt(workflow)
        print(f"Command received successfully! Task queue ID: {response['prompt_id']}")
    except Exception as e:
        print(f"Sending failed. Please check if ComfyUI is running in the background. Error message: {e}")


if __name__ == "__main__":
    main()