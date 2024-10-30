from flask import Blueprint, render_template

function = Blueprint('function', __name__)

@function.route('/function')
def function_view():
    return render_template('function.html')

@function.route('/function/pred')
def pred():
    return render_template('pred.html')