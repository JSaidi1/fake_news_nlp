import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from dataclasses import dataclass

CURRENT_FILE_PATH = Path(os.path.abspath(__file__))
BASE_DIR = CURRENT_FILE_PATH.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"
ENV_FILE_EXAMPLE_PATH = BASE_DIR / ".env.example"


# -------------------------------------------------------------------------------------------------------
#                                               LOAD ENV.
# -------------------------------------------------------------------------------------------------------

if os.path.exists(ENV_FILE_PATH):
    load_dotenv()
elif os.path.exists(ENV_FILE_EXAMPLE_PATH):
    load_dotenv(dotenv_path=ENV_FILE_EXAMPLE_PATH)
else:
    # logging.error(f"No .env or .env.example files were found in '{BASE_DIR}'")
    raise FileNotFoundError(f"No environment file (.env or .env.example) file was found in '{BASE_DIR}'")

# -------------------------------------------------------------------------------------------------------
#                                              USEFUL FNCS.
# -------------------------------------------------------------------------------------------------------
def str2bool(input_str: str) -> bool | None:
    if input_str.lower() in ["true", "yes", "y", "1"]:
        return True
    elif input_str.lower() in ["false", "no", "n", "0"]:
        return False
    else:
        return None
    
# =======================================================================================================
#                                            MINIO CONFIG.
# =======================================================================================================
@dataclass
class ApiConfig:
    api_title: str = os.getenv("TITLE")
    api_version: str = os.getenv("VERSION")
    api_description: str = os.getenv("DESCRIPTION")

    api_user: str = os.getenv("USER")
    api_password: str = os.getenv("PASSWORD")
    api_key: str = os.getenv("X_API_Key")

    api_ml_model_name: str = os.getenv("MODEL_NAME")
    api_ml_vectorizer_name: str = os.getenv("VECTORIZER_NAME")

@dataclass
class LogConfig:
    file_log: bool = str2bool(os.getenv("FILE_LOG"))
    log_dir_name: str = os.getenv("LOG_DIR_NAME")
    api_log_file_name: str = os.getenv("API_LOG_FILE_NAME")
    debug_mode: bool = str2bool(os.getenv("DEBUG_MODE"))

# -------------------------------------------------------------------------------------------------------
#                                     CREATE SINGLETON CFG. OBJECT
# -------------------------------------------------------------------------------------------------------
log_cfg = LogConfig()
api_cfg = ApiConfig()



# -------------------------------------------------------------------------------------------------------
#                                               CHECK
# -------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(log_cfg)
    print(api_cfg)