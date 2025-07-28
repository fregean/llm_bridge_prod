import hydra
from hle_benchmark import run_judge_results

@hydra.main(config_name="config_deepseek_api_for_tts", version_base=None, config_path="conf")
def main(cfg):
    run_judge_results.main(cfg)

if __name__ == "__main__":
    main()