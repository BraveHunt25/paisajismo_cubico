import cv2
import os
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Rutas de archivos de video
VIDEO_BOSQUE_ABEDUL = 'src/assets/original/bosque_abedul.mp4'
VIDEO_BOSQUE_CEREZO = 'src/assets/original/bosque_cerezo.mp4'
VIDEO_BOSQUE_HELADO = 'src/assets/original/bosque_helado.mp4'
VIDEO_BOSQUE_PALIDO = 'src/assets/original/bosque_palido.mp4'
VIDEO_JUNGLA = 'src/assets/original/jungla.mp4'

# Ruta de guardado por frame
FRAMES_BOSQUE_ABEDUL = 'src/assets/frames/bosque_abedul'
FRAMES_BOSQUE_CEREZO = 'src/assets/frames/bosque_cerezo'
FRAMES_BOSQUE_HELADO = 'src/assets/frames/bosque_helado'
FRAMES_BOSQUE_PALIDO = 'src/assets/frames/bosque_palido'
FRAMES_JUNGLA = 'src/assets/frames/jungla'

VIDEO_PATHS = {
    'bosque_abedul': (VIDEO_BOSQUE_ABEDUL, FRAMES_BOSQUE_ABEDUL),
    'bosque_cerezo': (VIDEO_BOSQUE_CEREZO, FRAMES_BOSQUE_CEREZO),
    'bosque_helado': (VIDEO_BOSQUE_HELADO, FRAMES_BOSQUE_HELADO),
    'bosque_palido': (VIDEO_BOSQUE_PALIDO, FRAMES_BOSQUE_PALIDO),
    'jungla': (VIDEO_JUNGLA, FRAMES_JUNGLA)
}

# Abrir el video
for video_name, (video_path, frames_path) in VIDEO_PATHS.items():
    try:
        cap = cv2.VideoCapture(video_path)
        os.makedirs(frames_path, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"FPS del video {video_name}: {fps:.3f}")
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("Se alcanzó el final del video")
                break
            
            # Tiempo (en segundos) estimado de este frame
            timestamp = frame_index / fps

            # Guardar frame
            filename = os.path.join(frames_path, f"frame_{frame_index:05d}_t{timestamp:.3f}.png")
            cv2.imwrite(filename, frame)

            frame_index += 1
        cap.release()
        logging.info(f"Extracción de frames de {video_name} completada.")
    except cv2.error as e:
        print(f"Error al abrir el video {video_name}: {e}")
    except Exception as e:
        print(f"Error al abrir el video {video_name}: {e}")
    finally:
        continue