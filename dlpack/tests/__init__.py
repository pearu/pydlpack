def run():
    import pytest, os

    pytest.main(["-svx", os.path.dirname(__file__)])
