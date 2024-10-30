from flask import Flask, url_for, render_template
from flask_bootstrap import Bootstrap5
from flask_sqlalchemy import SQLAlchemy
from .train import train
from .predict import predict

bootstrap = Bootstrap5()
database = SQLAlchemy()

def make_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Y:/Learning material/Fianl Year Project/Web/Web/database.db'
 
    bootstrap.init_app(app)
    database.init_app(app)

    @app.route("/")
    def home():
        return render_template("home.html")

    @app.teardown_request
    def teardown_request(exception):
        database.session.remove()

    with app.app_context():
        from .blueprints.function import function
        app.register_blueprint(function)
        import apis
        app.register_blueprint(apis.api_v1, url_prefix="/api")
        app.register_blueprint(train,url_prefix="/train")
        app.register_blueprint(predict,url_prefix="/predict")

        @app.route("/api")
        def api():
            return url_for("api")

    return app