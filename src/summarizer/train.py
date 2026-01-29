import art
import asyncio
from dotenv import load_dotenv
import random
from art.serverless.backend import ServerlessBackend
import weave

from summarizer import rollout, SummarizerScenario
from summarizer.load_documents import load_documents

load_dotenv()

AGENT_NAME = "agent-000"
PROJECT_NAME = "summarize"


async def main():
    weave.init(project_name=PROJECT_NAME)
    val_documents, train_documents = load_documents()

    backend = ServerlessBackend()

    model = art.TrainableModel(
        name=AGENT_NAME,
        project=PROJECT_NAME,
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    await model.register(backend)

    batch_size = 10  # Process this many documents per batch
    num_epochs = 1  # Number of complete passes through the training data

    start_step = await model.get_step()
    max_steps = 1000

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_documents)

        # Calculate how many batches we can process in this epoch
        num_batches = min(
            len(train_documents) // batch_size, (max_steps - start_step) // num_epochs
        )

        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break

            print(
                f"Epoch {epoch + 1}, Batch {batch + 1}/{num_batches}, Step {current_step}"
            )

            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size

            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(
                                model,
                                SummarizerScenario(doc=document, step=current_step),
                            )
                            for _ in range(2)
                        )
                        for document in val_documents
                    ),
                    pbar_desc=f"gather val (epoch {epoch + 1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(model, SummarizerScenario(doc=document))
                            for _ in range(10)
                        )
                        for document in train_documents[batch_start_idx:batch_end_idx]
                    ),
                    pbar_desc=f"gather train (epoch {epoch + 1}, batch {batch + 1})",
                ),
            )

            await model.log(val_groups)
            await model.delete_checkpoints()
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )


if __name__ == "__main__":
    asyncio.run(main())
