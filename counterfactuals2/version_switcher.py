import transformers
import pip
import time
import subprocess


def install_transformers(version: str):
    pip.main(['uninstall', "-y", "transformers", "tokenizers"])
    print("uninstalled transformers, tokenizers")
    print("sleeping for some time")
    time.sleep(20)
    pip.main(['install', "transformers==" + version])
    time.sleep(20)
    print("#####################################\n" * 2)
    print("installed version", version)
    print("#####################################\n" * 2)


def evaluate():
    print("executing subprocess")
    subprocess.run(["D:\\A_Uni\\A_MasterThesis\\MMD\\venv\\Scripts\\python.exe", "auto_evaluator.py"])
    print("subprocess terminated")


try:
    print("Starting with transformers.__version__", transformers.__version__)
    if transformers.__version__ == "4.17.0":
        evaluate()
        install_transformers("4.37.0")
        evaluate()
    else:
        evaluate()
        install_transformers("4.17.0")
        evaluate()
except:
    print("no transformers installed")
    install_transformers("4.17.0")
    evaluate()
    install_transformers("4.37.0")
    evaluate()
