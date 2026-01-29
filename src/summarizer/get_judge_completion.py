from async_lru import alru_cache
import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

semaphore = asyncio.Semaphore(20)


client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


@alru_cache(maxsize=1024)
async def get_judge_completion(
    prompt, temperature=0.0, max_tokens=600, retries=3, timeout=10
) -> str:
    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="google/gemini-3-flash-preview",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries:
                print(
                    f"[Retry {attempt}/{retries}] get_judge_completion failed: {e}. Retrying..."
                )
                await asyncio.sleep(3)
            else:
                print(
                    f"[Failure] get_judge_completion failed after {retries} attempts: {e}"
                )
                return "ERROR: Get judge completion failed"


def clear_judge_cache():
    """Clear the cache for get_judge_completion."""
    get_judge_completion.cache_clear()
    print("Judge cache cleared")
