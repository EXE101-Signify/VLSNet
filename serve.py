"""Inference-only entry point for the VSLNet container."""
import os
from types import SimpleNamespace

import uvicorn
import yaml

from mode.inferencer import create_app


def main():
    with open('config/vsl_net_configuration.yaml', encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)['BASIC']
    app = create_app(SimpleNamespace(**settings))
    uvicorn.run(app, host=os.getenv('VSLNET_HOST', '127.0.0.1'),
                port=int(os.getenv('VSLNET_PORT', '8001')), ws_max_size=256_000)


if __name__ == '__main__':
    main()
