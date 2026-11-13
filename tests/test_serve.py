import os
import unittest
from unittest.mock import patch

import serve


class DeploymentEntryPointTests(unittest.TestCase):
    def test_container_entry_point_uses_configured_bind_address(self):
        with patch.dict(os.environ, {'VSLNET_HOST': '0.0.0.0', 'VSLNET_PORT': '8001'}), \
                patch.object(serve, 'create_app', return_value=object()) as create_app, \
                patch.object(serve.uvicorn, 'run') as run:
            serve.main()
        create_app.assert_called_once()
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs['host'], '0.0.0.0')
        self.assertEqual(run.call_args.kwargs['port'], 8001)
        self.assertEqual(run.call_args.kwargs['ws_max_size'], 256_000)


if __name__ == '__main__':
    unittest.main()
