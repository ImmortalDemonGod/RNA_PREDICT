import subprocess

def test_dssr_installation(dssr_path="x3dna-dssr"):
    """Test if X3DNA-DSSR is installed and working correctly."""
    try:
        result = subprocess.run(
            [dssr_path, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

def test_dssr_installation_success(monkeypatch):
    class DummyCompleted:
        stdout = "DSSR v2.4.3"
        stderr = ""
        returncode = 0
    def dummy_run(*args, **kwargs):
        return DummyCompleted()
    monkeypatch.setattr(subprocess, "run", dummy_run)
    assert test_dssr_installation("dummy-dssr") is True

def test_dssr_installation_not_found(monkeypatch):
    def dummy_run(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", dummy_run)
    assert test_dssr_installation("dummy-dssr") is False

def test_dssr_installation_calledprocesserror(monkeypatch):
    class DummyError(subprocess.CalledProcessError):
        def __init__(self):
            super().__init__(1, 'dummy')
            self.stderr = "error"
    def dummy_run(*args, **kwargs):
        raise DummyError()
    monkeypatch.setattr(subprocess, "run", dummy_run)
    assert test_dssr_installation("dummy-dssr") is False
