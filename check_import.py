import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"), flush=True)
print("PATH:", os.environ.get("PATH"), flush=True)
import domainbed.scripts.sweep
print("DomainBed module imported successfully", flush=True)
import sys
print("sys.path:", sys.path, flush=True)
sys.exit(0)
