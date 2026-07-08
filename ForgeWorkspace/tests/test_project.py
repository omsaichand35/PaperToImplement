from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import json

from ForgeWorkspace.workspace.project import (
    initialize_project
)

from ForgeWorkspace.workspace.inspector import (
    inspect_project
)


result = initialize_project(
    project_name="transformer_reproduction",
    framework="pytorch",
    task_type="sequence_modeling"
)


print(
    "\nInitialization Result:\n"
)

print(
    json.dumps(
        result,
        indent=2
    )
)


inspection = inspect_project(
    "transformer_reproduction"
)


print(
    "\nProject Tree:\n"
)

print(
    inspection["tree"]
)