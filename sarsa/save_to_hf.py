from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json
import pickle 
from eval import *
from record import *

def push_to_hub(repo_id, 
                model,
                env,
                video_fps=4,
                local_repo_path="/home/pi/reinforcement_learning/sarsa/",
                commit_message="Push Q-Learning agent to Hub",
                token= None, env_id=None
                ):

    _, repo_name = repo_id.split("/")

    eval_env = env
    
    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    
    repo_url = api.create_repo(
            repo_id=repo_id,
            token=token,
            private=False,
            exist_ok=True,)
    
    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=True)
    repo.git_pull()
    
    repo.lfs_track(["*.mp4"])

    # Step 1: Save the model
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") == False:
            model["slippery"] = False

    print(model)
    
        
    # Pickle the model
    with open(Path(repo_local_path)/'q-learning.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Step 2: Evaluate the model and build JSON
    mean_reward, std_reward = eval_model(eval_env, model["n_eval_episodes"], model["qtable"], model["eval_seed"])

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
            "env_id": model["env_id"], 
            "mean_reward": mean_reward,
            "n_eval_episodes": model["n_eval_episodes"],
            "eval_datetime": eval_form_datetime,
    }
    # Write a JSON file
    with open(Path(repo_local_path) / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 3: Create the model card
    # Env id
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [
            env_name,
            "q-learning",
            "reinforcement-learning",
            "custom-implementation"
        ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
        )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
    # **Q-Learning** Agent playing **{env_id}**
    This is a trained model of a **Q-Learning** agent playing **{env_id}** .
    """

    model_card += """
    ## Usage
    ```python
    """

    model_card += f"""model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_id"])

    evaluate_agent(env, model["n_eval_episodes"], model["qtable"], model["eval_seed"])
    """

    model_card +="""
    ```
    """

    readme_path = repo_local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 4: Record a video
    video_path =  repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)
    
    # Push everything to hub
    print(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message=commit_message)

    print(f"Your model is pushed to the hub. You can view your model here: {repo_url}")
        