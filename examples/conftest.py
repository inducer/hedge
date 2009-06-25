# This file tells py.test to scan for test_xxx() functions in
# any file below here, not just those named test_xxxx.py.

def pytest_collect_file(path, parent):
    if "hedge/examples" in str(path.dirpath()) and path.ext == ".py":
        return parent.Module(path, parent=parent)
