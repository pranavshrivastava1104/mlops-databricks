from pydantic import BaseModel
import yaml

class ProjectConfig(BaseModel):
    catalog_name:str
    schema_name:str
    target:str
    experiment_name:str
    parameters:dict
    num_features:list[str]
    cat_features:list[str]

    @classmethod
    def from_yaml(cls,config_path:str,env:str):
        if env not in ("dev","prod",acc):
            raise ValueError("invalid envirnoment")
        with open(config_path,"r") as f:
            raw=yaml.safe_load(f)
        env_block=raw[env]
        shared={
            k:v for k,v in raw.items()
            if k not in ("dev","prod","acc")
        }
        merged={**shared,**env_block}

        return cls(**merged)


