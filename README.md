# Proyecto Final de Visión Artificial
- Hernández Jiménez Erick Yael
- Mazariegos Aguilar Julio Darikson
- Chávez Torreblanca Angel Alexis

**Última modificación: 21/06/2025 11:47 AM**
_Actualizar en cada avance que se tenga_

## Preparación
El repositorio ya se encuentra dividido en las carpetas necesarias para que los scripts funcionen (aunque, dependiendo del intérprete que usen, las cadenas de caracteres que representan la ruta relativa de los archivos originales y las carpetas puedan variar). En cualquier caso, recomiendo que trabajen desde el directorio raíz del proyecto que, en el prompt del CMD, se debe ver algo así: `C:\Ruta\hacia\el\proyecto\Proyecto Final>`.

> [!WARNING]
> Debido a que el archivo supera los 100GB que puede manejar Git, es necesario la extensión Git for Large Files Storage, esto se hace desde el CMD con el comando: `git lfs install`. Se debe ejecutar en cuando se conecten al repositorio y antes de ejecutar "pull" a la rama _main_.

Para asegurar la replicación del proyecto, sugiero seguir los siguientes pasos:
1. Crear un entorno virtual desde la raíz del proyecto ejcutando: `python -m venv nombre_del_entorno`, donde *nombre_del_entorno* es el nombre que se le desee dar. El nombre *.venv* se sugiere por defecto.
2. Activar el entorno virtual con el comando: `nombre_del_entorno\Scripts\activate`.
3. Descargar las librerías necesarias en todo el proyexto con: `pip install -r requirements.txt`. El archivo [requirements.txt](requirements.txt) no se debe mover nunca de la dirección raíz. En caso de agregar dependencias o modificarlas, ejecuten el comando: `pip install -r requirements.txt`.
4. Descomprimir los archivos en la [caperta ZIP](src/assets/original.zip) con los videos originales.
5. Ejecutar el [script](src/treatment/frame_extraction.py). (Asegúrense de verificar que los resultados queden en el directorio **src/assets/frames/_nombre del bioma_**).

## Progreso
En este apartado se indican los progresos seccionados en los biomas seleccionados, base indexada, análisis de los datos, desarrollo de clasificador, desarrollo de sintetizador de voz.

### Biomas seleccionados
Los videos corresponden a los biomas:
- Bosque de abedúl
- Bosque de cerezo
- Bosque helado con montaña nevada
- Bosque pálido
- Jungla

### Base indexada
Aún no se crea la base indexada, únicamente se han dividido las carpetas para mantener congruencia en la relación de los frames con los videos. **Queda pendiente la creación de la base indexada**.

### Análisis de datos
Se ha probado con la implementación rápida de _k-means_ de _scikit-learn_, obteniendo que, para obtener resultados aceptables que no mezclen los colores en un promedio muy general, el **número de centroides debe ser mayor a 5**. Pero, estos resultados aún mezclan distintas características en una mezcla variada de colores. Probar con otros métodos se sugiere para seguir explorando.

### Desarrollo de clasificador
Aún no se supera el análisis de datos, por lo que **queda pendiente el desarrollo del clasificador**.

### Desarrollo de sintetizador de voz
Aún no se ha analizado el desarrollo del sintentizador de voz. **Queda pendiente el desarrollo del sintetizador de voz**.
