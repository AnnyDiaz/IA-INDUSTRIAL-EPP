from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from flask import jsonify
import os
import uuid
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from werkzeug.utils import secure_filename
from transformers import pipeline

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Configuración inicial
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Carga modelo YOLO para detección de EPP
model = YOLO("ppe.pt")
print("Etiquetas detectables:", model.names)

# Mapeo de etiquetas a nombres legibles y requisitos normativos

EPP_DEFINICION = {
    "Hardhat": {
        "nombre": "Casco de seguridad",
        "normativa": "Norma ANSI Z89.1 / OSHA 1910.135",
        "descripcion": "Protección contra impactos en la cabeza"
    },
    
   "Safety Vest": {
       "nombre": "Chaleco reflectante",
       "normativa": "Norma ANSI/ISEA 107 / OSHA 1910.132",
       "descripcion": "Visibilidad en áreas con movimiento de vehículos"
    },
    #"Gloves": {
     #   "nombre": "Guantes de seguridad",
      #  "normativa": "Norma OSHA 1910.138",
      #  "descripcion": "Protección de manos contra riesgos mecánicos o químicos"
   # },
    "Mask": {
        "nombre": "Mascarilla de protección",
        "normativa": "Norma NIOSH / OSHA 1910.134",
        "descripcion": "Protección respiratoria contra polvo o sustancias peligrosas"
    },
    "Person":{
        "nombre": "Personas",
        "descripcion": "Todos los individuos de la especie humana, cualquiera que sea su edad, sexo, estirpe o condición"
    
        
    }
    
}

EPP_REQUERIDOS = list(EPP_DEFINICION.keys())
ultima_deteccion = []
contexto_chat = []

# Carga modelo de diálogo
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model_chat = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    print("Modelo de chat cargado correctamente")
except Exception as e:
    print(f"Error al cargar modelo de chat: {str(e)}")
    tokenizer = None
    model_chat = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analizar_imagen(imagen_path):
    global ultima_deteccion
    
    try:
        results = model(imagen_path)
        clases_detectadas = results[0].boxes.cls
        etiquetas_detectadas = [results[0].names[int(clase)] for clase in clases_detectadas]
        confianzas = [float(conf) for conf in results[0].boxes.conf]
        
        print(f"Detecciones: {list(zip(etiquetas_detectadas, confianzas))}")
        
        # Filtrar solo EPPs relevantes con confianza > 50%
        detecciones_reales = [
            etiq for etiq, conf in zip(etiquetas_detectadas, confianzas) 
            if etiq in EPP_REQUERIDOS and conf > 0.5
        ]
        
        ultima_deteccion = list(set(detecciones_reales))
        
        # Generar informe detallado
        informe = {
            "detectados": [EPP_DEFINICION[e] for e in ultima_deteccion],
            "faltantes": [EPP_DEFINICION[e] for e in EPP_REQUERIDOS if e not in ultima_deteccion],
            "cumplimiento": len(ultima_deteccion) == len(EPP_REQUERIDOS),
            "descripciones": [EPP_DEFINICION[e]["descripcion"] for e in EPP_REQUERIDOS if e not in ultima_deteccion]
        }
        
        # Guardar imagen con anotaciones
        filename = f"resultado_{uuid.uuid4().hex}.jpg"
        salida = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        annotated_img = results[0].plot()
        cv2.imwrite(salida, annotated_img)
        
        return informe, filename
    
    except Exception as e:
        print(f"Error en análisis de imagen: {str(e)}")
        return None, None

def generar_respuesta_chat(pregunta):
    if not ultima_deteccion:
        return "Primero debes subir una imagen para analizar el equipo de protección."

    pregunta = pregunta.lower()

    # Diccionario de sinónimos y sus claves de detección
    sinonimos_epp = {
        "Hardhat": ["casco", "helmet", "sombrero duro"],
        "Safety Vest": ["chaleco", "chaleco reflectante", "safety vest", "chaleco de seguridad"],
        "Glasses": ["gafas", "lentes", "anteojos", "protección ocular"],
        "Mask": ["tapabocas", "mascarilla", "cubrebocas", "barbijo", "nasobuco"],
        "Gloves": ["guantes", "protección de manos", "protección manual", "glove", "manoplas"],
        "Person": ["personas", "individuo", "hombre", "mujer", "humano"]
    }

    detectados = [EPP_DEFINICION[e]["nombre"].lower() for e in ultima_deteccion if e in EPP_DEFINICION]
    faltantes = [EPP_DEFINICION[e]["nombre"].lower() for e in EPP_REQUERIDOS if e not in ultima_deteccion]

    # Verificación si hay persona sin EPP completo
    if "Person" in ultima_deteccion and any(epp not in ultima_deteccion for epp in EPP_REQUERIDOS):
        return "Se ha detectado una persona que no cumple con los elementos de protección personal requeridos según la normativa de seguridad industrial."

    # Revisión por EPP individual
    for clave, lista_sinonimos in sinonimos_epp.items():
        if any(p in pregunta for p in lista_sinonimos):
            if clave in ultima_deteccion:
                return f"Sí, el trabajador está usando {EPP_DEFINICION[clave]['nombre'].lower()}."
            else:
                return f"No, el trabajador no está usando {EPP_DEFINICION[clave]['nombre'].lower()}."

    # Preguntas sobre cumplimiento
    if any(p in pregunta for p in ["qué le falta", "faltan", "le falta algo", "le faltan"]):
        if not faltantes:
            return "No faltan elementos de protección. El trabajador cumple con todos los EPP requeridos."
        else:
            return "Faltan los siguientes elementos de protección: " + ", ".join(faltantes) + "."

    if any(p in pregunta for p in ["cumple", "normativa", "reglamento", "seguridad"]):
        return "Sí, cumple con todos los EPP requeridos." if not faltantes else "No, no cumple con todos los EPP requeridos."

    if any(p in pregunta for p in ["qué está usando", "qué tiene puesto", "qué lleva puesto", "qué protección tiene"]):
        return "El trabajador está usando: " + ", ".join(detectados) + "."

    # Respuesta general
    return (
        "Puedo responder preguntas como: "
        "'¿Está usando casco?', '¿Faltan elementos de seguridad?', "
        "'¿Cumple con la normativa?', '¿Qué le falta?', entre otras."
    )


    try:
        respuesta = qa_model(question=pregunta, context=contexto)
        return respuesta['answer']
    except Exception as e:
        return f"Error al responder: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagen' in request.files:
            file = request.files['imagen']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{uuid.uuid4().hex}.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                informe, img_resultado = analizar_imagen(filepath)
                
                if informe:
                    cumplimiento = "✅ CUMPLE CON LOS EPP REQUERIDOS" if informe["cumplimiento"] else "❌ NO CUMPLE CON TODOS LOS EPP"
                    return jsonify({
                        "status": "success",
                        "data": {
                            "cumplimiento": cumplimiento,
                            "detectados": [e["nombre"] for e in informe["detectados"]],
                            "faltantes": [e["nombre"] for e in informe["faltantes"]],
                            "normativas": [e["normativa"] for e in informe["faltantes"]],
                            "descripciones": [e["descripcion"] for e in informe["faltantes"]],
                            "imagen": img_resultado
                        }
                    })
        
        elif 'pregunta' in request.form:
            pregunta = request.form['pregunta'].strip()
            
            # Filtrar preguntas no relevantes
            palabras_clave = ['epp', 'seguridad', 'casco', 'chaleco', 'gafas', 'protección', 'norma', 'osha', 'ansi']
            if not any(palabra in pregunta.lower() for palabra in palabras_clave):
                return jsonify({
                    "status": "success",
                    "respuesta": "Solo puedo responder preguntas sobre equipos de protección personal (EPP) y seguridad industrial."
                })
            
            respuesta = generar_respuesta_chat(pregunta)
            return jsonify({"status": "success", "respuesta": respuesta.capitalize()})
    
    return render_template('index.html')


@app.route('/analyze', methods=['GET','POST'])
def analyze():
    if 'imagen' not in request.files:
        return jsonify({"status": "error", "message": "No se envió ninguna imagen."})

    image = request.files['imagen']
    if image.filename == '':
        return jsonify({"status": "error", "message": "Nombre de archivo vacío."})

    if image:
        filename = str(uuid.uuid4()) + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        informe, img_resultado = analizar_imagen(filepath)

        if informe:
            cumplimiento = "✅ CUMPLE CON LOS EPP REQUERIDOS" if informe["cumplimiento"] else "❌ NO CUMPLE CON TODOS LOS EPP"
            return jsonify({
                "status": "success",
                "data": {
                    "cumplimiento": cumplimiento,
                    "detectados": [e["nombre"] for e in informe["detectados"]],
                    "faltantes": [e["nombre"] for e in informe["faltantes"]],
                    "normativas": [e["normativa"] for e in informe["faltantes"]],
                    "descripciones": [e["descripcion"] for e in informe["faltantes"]],
                    "imagen": img_resultado
                }
            })

    return jsonify({"status": "error", "message": "Error al procesar la imagen."})
@app.route('/ask', methods=['POST'])
def ask():
    pregunta = request.form.get('pregunta')
    if not pregunta:
        return jsonify({"status": "error", "message": "No se recibió ninguna pregunta."})

    respuesta = generar_respuesta_chat(pregunta)
    return jsonify({"status": "success", "respuesta": respuesta})

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)