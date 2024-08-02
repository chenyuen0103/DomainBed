import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("PATH:", os.environ.get("PATH"))
import domainbed.scripts.sweep
print("DomainBed module imported successfully")
import sys
print("sys.path:", sys.path)
sys.exit(0)
