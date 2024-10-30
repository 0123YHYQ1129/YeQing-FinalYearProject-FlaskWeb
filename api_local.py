from flask import Flask
from datetime import timedelta

app = Flask(__name__)

import apis

app.register_blueprint(apis.api_v1, url_prefix="/")


if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001,
        host="0.0.0.0",
        threaded=True,
    )