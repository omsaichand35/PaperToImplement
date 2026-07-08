import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\omsai\PycharmProjects\PaperForge\ForgeCode')

from codeforge.planner import create_implementation_plan
from tests.test_orchestrator import SPEC

async def test():
    plan = await create_implementation_plan(SPEC)
    out_path = Path(r'C:\Users\omsai\PycharmProjects\PaperForge\ForgeCode\scratch\out.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=== SUCCESSFUL NORMALIZED PLAN ===\n")
        f.write(json.dumps(plan, indent=2, ensure_ascii=False))
    print("Done writing to out.txt")

if __name__ == "__main__":
    asyncio.run(test())
