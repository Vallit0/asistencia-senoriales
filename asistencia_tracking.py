import cv2
import numpy as np
import json
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# Configuración
DB_FILE = '/app/empleados.json'
LOGS_FILE = '/app/asistencia_log.json'
THRESHOLD = 0.4

# Zonas (ajustar según tu cámara)
ZONA_IZQUIERDA = 150   # X menor a esto = zona izquierda (oficinas)
ZONA_DERECHA = 490     # X mayor a esto = zona derecha (oficinas)
# Entre ZONA_IZQUIERDA y ZONA_DERECHA = zona centro (puerta)

print("Cargando modelo con GPU...")
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Modelo cargado!")

tracks = {}
next_id = 0

def cargar_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            for emp in data:
                emp['embedding'] = np.array(emp['embedding'])
            return data
    return []

def guardar_db(empleados):
    data = [{'nombre': e['nombre'], 'embedding': e['embedding'].tolist()} for e in empleados]
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def registrar_log(nombre, tipo):
    log = {'nombre': nombre, 'tipo': tipo, 'timestamp': datetime.now().isoformat()}
    logs = []
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            logs = json.load(f)
    logs.append(log)
    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tipo}: {nombre}")

def comparar_embedding(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def reconocer(embedding, empleados):
    mejor_match, mejor_score = None, 0
    for emp in empleados:
        score = comparar_embedding(embedding, emp['embedding'])
        if score > mejor_score:
            mejor_score, mejor_match = score, emp['nombre']
    return (mejor_match, mejor_score) if mejor_score > THRESHOLD else (None, mejor_score)

def get_zona(x):
    if x < ZONA_IZQUIERDA:
        return 'izquierda'
    elif x > ZONA_DERECHA:
        return 'derecha'
    else:
        return 'centro'

def find_track(nombre):
    global next_id
    
    for tid, track in tracks.items():
        if track['nombre'] == nombre and not track['cruzado']:
            return tid
    
    tid = next_id
    next_id += 1
    tracks[tid] = {
        'nombre': nombre,
        'zona_inicial': None,
        'zona_actual': None,
        'cruzado': False
    }
    return tid

def check_cruce(tid, centro_x):
    track = tracks[tid]
    
    if track['cruzado']:
        return None
    
    zona_actual = get_zona(centro_x)
    
    # Primera vez que vemos a esta persona
    if track['zona_inicial'] is None:
        track['zona_inicial'] = zona_actual
        track['zona_actual'] = zona_actual
        return None
    
    track['zona_actual'] = zona_actual
    
    # Verificar transición
    zona_inicial = track['zona_inicial']
    
    # ENTRADA: viene del centro (puerta) y va a izquierda o derecha (oficinas)
    if zona_inicial == 'centro' and zona_actual in ['izquierda', 'derecha']:
        track['cruzado'] = True
        return 'ENTRADA'
    
    # SALIDA: viene de izquierda o derecha (oficinas) y va al centro (puerta)
    if zona_inicial in ['izquierda', 'derecha'] and zona_actual == 'centro':
        track['cruzado'] = True
        return 'SALIDA'
    
    return None

def limpiar_tracks():
    to_delete = [tid for tid, t in tracks.items() if t['cruzado']]
    for tid in to_delete:
        del tracks[tid]

def main():
    global ZONA_IZQUIERDA, ZONA_DERECHA
    
    empleados = cargar_db()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Empleados: {len(empleados)}")
    print("Controles:")
    print("  R = Registrar")
    print("  Q = Salir")
    print("  1/2 = Mover línea izquierda")
    print("  3/4 = Mover línea derecha")
    
    prev_time = datetime.now()
    entradas_hoy = 0
    salidas_hoy = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        faces = app.get(frame)
        
        now = datetime.now()
        fps = 1 / max((now - prev_time).total_seconds(), 0.001)
        prev_time = now
        
        for face in faces:
            box = face.bbox.astype(int)
            centro_x = (box[0] + box[2]) // 2
            centro_y = (box[1] + box[3]) // 2
            
            nombre, score = reconocer(face.embedding, empleados)
            
            if nombre:
                color = (0, 255, 0)
                label = f"{nombre} ({score:.2f})"
                
                tid = find_track(nombre)
                cruce = check_cruce(tid, centro_x)
                
                if cruce:
                    registrar_log(nombre, cruce)
                    if cruce == 'ENTRADA':
                        entradas_hoy += 1
                    else:
                        salidas_hoy += 1
            else:
                color = (0, 0, 255)
                label = "Desconocido"
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (centro_x, centro_y), 5, color, -1)
        
        # Dibujar líneas verticales
        cv2.line(frame, (ZONA_IZQUIERDA, 0), (ZONA_IZQUIERDA, 480), (255, 255, 0), 2)
        cv2.line(frame, (ZONA_DERECHA, 0), (ZONA_DERECHA, 480), (255, 255, 0), 2)
        
        # Etiquetas de zonas
        cv2.putText(frame, "OFICINAS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "PUERTA", (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "OFICINAS", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Entradas: {entradas_hoy} | Salidas: {salidas_hoy}", (200, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Empleados: {len(empleados)}", (500, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Sistema de Asistencia', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            nombre = input("Nombre: ").strip()
            if nombre:
                ret, frame = cap.read()
                faces = app.get(frame)
                if len(faces) == 1:
                    empleados.append({'nombre': nombre, 'embedding': faces[0].embedding})
                    guardar_db(empleados)
                    print(f"✓ {nombre} registrado!")
                else:
                    print(f"Error: {len(faces)} caras detectadas")
        elif key == ord('1'):
            ZONA_IZQUIERDA = max(10, ZONA_IZQUIERDA - 10)
            print(f"Línea izq: X={ZONA_IZQUIERDA}")
        elif key == ord('2'):
            ZONA_IZQUIERDA = min(300, ZONA_IZQUIERDA + 10)
            print(f"Línea izq: X={ZONA_IZQUIERDA}")
        elif key == ord('3'):
            ZONA_DERECHA = max(340, ZONA_DERECHA - 10)
            print(f"Línea der: X={ZONA_DERECHA}")
        elif key == ord('4'):
            ZONA_DERECHA = min(630, ZONA_DERECHA + 10)
            print(f"Línea der: X={ZONA_DERECHA}")
        
        if frame_count % 100 == 0:
            limpiar_tracks()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
