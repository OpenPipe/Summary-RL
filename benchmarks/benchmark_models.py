import os
import asyncio
import art
from art.serverless.backend import ServerlessBackend
from summarizer.get_judge_completion import clear_judge_cache
from summarizer.load_documents import load_documents
from summarizer.rollout import rollout, SummarizerScenario
from summarizer.train import CLUSTER_NAME, PROJECT_NAME


gpt_4o = art.Model(
    name="gpt-4o",
    project=PROJECT_NAME,
    inference_model_name="openai/gpt-4o",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

gpt_4_1 = art.Model(
    name="gpt-4.1",
    project=PROJECT_NAME,
    inference_model_name="openai/gpt-4.1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

gemini_2_5_pro = art.Model(
    name="gemini-2.5-pro",
    project=PROJECT_NAME,
    inference_model_name="google/gemini-2.5-pro-preview",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

sonnet_4 = art.Model(
    name="sonnet-4",
    project=PROJECT_NAME,
    inference_model_name="anthropic/claude-sonnet-4",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

val_documents, _ = load_documents()


async def benchmark_model(model: art.Model) -> None:
    trajectory_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(model, SummarizerScenario(doc=document)) for _ in range(2)
            )
            for document in val_documents
        ),
        pbar_desc=model.name,
    )
    await model.log(trajectories=trajectory_groups, split="val")


async def main():
    # clear_judge_cache()
    backend = ServerlessBackend()
    models = [
        gpt_4o,
        gpt_4_1,
        gemini_2_5_pro,
        sonnet_4,
    ]
    for model in models:
        await model.register(backend)
    # benchmark all models simultaneously
    await asyncio.gather(*[benchmark_model(model) for model in models])

    for model in models:
        await backend._experimental_push_to_s3(model)


if __name__ == "__main__":
    asyncio.run(main())
