from app import app
from flask import render_template, request
from app.generacion import semillas_generacion, generar
import json

@app.route('/')
@app.route('/index')
@app.route('/generacion')
def ini_web_generacion():
	semillas, sem_random, idxs = semillas_generacion()
	return render_template('generacion.html', semillas= semillas, sem_random=sem_random, idxs=idxs)

@app.route('/refrescar_semillas', methods=['GET'])
def refrescar_semillas():
	semillas, sem_random, idxs = semillas_generacion()
	return json.dumps({'status':'OK','semillas':semillas,'sem_random':sem_random, 'idxs':idxs})

@app.route('/generar_texto', methods=['GET'])
def generar_texto():
	if 'div' in request.args and 'np' in request.args:
		info, texto_generado = generar(request.args.get('idx', 0, type=int), diversidad = request.args.get('div', 0, type=float), num_palabras= request.args.get('np', 0, type=int))
	elif 'div' in request.args:
		info, texto_generado = generar(request.args.get('idx', 0, type=int), diversidad = request.args.get('div', 0, type=float))
	elif 'np' in request.args:
		info, texto_generado = generar(request.args.get('idx', 0, type=int), num_palabras = request.args.get('np', 0, type=int))
	else:
		info, texto_generado = generar(request.args.get('idx', 0, type=int))
	return json.dumps({'status':'OK','info':info,'texto_generado':texto_generado})
