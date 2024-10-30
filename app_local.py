from app import make_app

flask_app = make_app()

if __name__ == "__main__":
    flask_app.run(
        debug=True,
        port=5010,
        host="0.0.0.0",
        threaded=True,
    )


