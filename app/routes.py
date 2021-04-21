from app import app
from flask import render_template
from app.generacion import devuelve_generacion_aleatoria

@app.route('/')
@app.route('/index')
def index():
	user = {'username': 'Bitch'}
	return render_template('index.html', title='Home', user=user)

@app.route('/generacion')
def generacion():
	info, texto_generado = devuelve_generacion_aleatoria()
	return render_template('generacion.html', info = info, texto_generado = texto_generado)