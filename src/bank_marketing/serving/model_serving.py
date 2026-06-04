from __future__ import annotations

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger

from bank_marketing.config import ProjectConfig


class ModelServing:
    """
    Handles Databricks Model Serving deployment for the bank subscription model.
    """

    def __init__(
        self,
        config: ProjectConfig,
        env: str,
        workload_size: str = "Small",
        scale_to_zero_enabled: bool = True,
    ) -> None:
        self.config = config
        self.env = env
        self.workload_size = workload_size
        self.scale_to_zero_enabled = scale_to_zero_enabled

        self.client = WorkspaceClient()
        self.model_name = (
            f"{self.config.catalog_name}.models.subscription_classifier"
        )
        self.endpoint_name = f"bank-subscription-classifier-{self.env}"

    def _build_served_entity(self, version: str) -> ServedEntityInput:
        return ServedEntityInput(
            entity_name=self.model_name,
            entity_version=str(version),
            workload_size=self.workload_size,
            scale_to_zero_enabled=self.scale_to_zero_enabled,
        )

    def _endpoint_exists(self) -> bool:
        try:
            self.client.serving_endpoints.get(name=self.endpoint_name)
            return True
        except NotFound:
            return False

    def _wait_until_endpoint_ready(
        self,
        timeout_seconds: int = 1200,
        poll_interval_seconds: int = 30,
    ) -> None:
        logger.info(
            f"Waiting for endpoint '{self.endpoint_name}' to become ready."
        )

        start_time = time.time()

        while True:
            endpoint = self.client.serving_endpoints.get(
                name=self.endpoint_name,
            )
            state = endpoint.state

            ready_state = str(getattr(state, "ready", "")).upper()
            config_update_state = str(
                getattr(state, "config_update", "")
            ).upper()

            logger.info(
                f"Endpoint state: ready={ready_state}, "
                f"config_update={config_update_state}"
            )

            is_not_updating = "NOT_UPDATING" in config_update_state
            is_ready = "READY" in ready_state or ready_state == ""

            if is_not_updating and is_ready:
                logger.info(f"Endpoint '{self.endpoint_name}' is ready.")
                return

            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Endpoint '{self.endpoint_name}' did not become ready "
                    f"within {timeout_seconds} seconds."
                )

            time.sleep(poll_interval_seconds)

    def deploy_or_update_serving_endpoint(self, version: str) -> None:
        if version is None or str(version).strip() == "":
            raise ValueError(
                "model version is empty. Cannot deploy serving endpoint."
            )

        version = str(version)

        logger.info(
            f"Preparing deployment. "
            f"endpoint={self.endpoint_name}, "
            f"model={self.model_name}, "
            f"version={version}"
        )

        served_entity = self._build_served_entity(version=version)

        if self._endpoint_exists():
            logger.info(
                f"Endpoint '{self.endpoint_name}' already exists. "
                f"Updating it to model version {version}."
            )
            self.client.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=[served_entity],
            )
        else:
            logger.info(
                f"Endpoint '{self.endpoint_name}' does not exist. "
                f"Creating it with model version {version}."
            )
            endpoint_config = EndpointCoreConfigInput(
                served_entities=[served_entity],
            )
            self.client.serving_endpoints.create(
                name=self.endpoint_name,
                config=endpoint_config,
            )

        self._wait_until_endpoint_ready()

        logger.info(
            f"Deployment completed successfully. "
            f"endpoint={self.endpoint_name}, "
            f"model={self.model_name}, "
            f"version={version}"
        )
