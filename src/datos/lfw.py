"""Carga de LFW: hasta 50 identidades con al menos 20 fotografías."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros

MIN_FOTOS_POR_PERSONA = 20
MAX_IDENTIDADES = 50


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo."""

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        # 1. Descargar LFW solo con personas que tienen >= 20 fotos.
        lfw = fetch_lfw_people(
            min_faces_per_person=MIN_FOTOS_POR_PERSONA,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        images = np.asarray(lfw.images)
        y_original = np.asarray(lfw.target, dtype=int)
        nombres_originales = np.asarray(lfw.target_names)

        # 2. Conservar como máximo 50 identidades, las de mayor frecuencia.
        #    El desempate por id mantiene el resultado reproducible.
        ids, conteos = np.unique(y_original, return_counts=True)
        orden = np.lexsort((ids, -conteos))
        elegidos = ids[orden][:MAX_IDENTIDADES]
        mascara = np.isin(y_original, elegidos)
        images = images[mascara]
        y_seleccion = y_original[mascara]

        # 3. Reindexar y a 0..C-1 y conservar los nombres de esas personas.
        ids_usados, y = np.unique(y_seleccion, return_inverse=True)
        y = np.asarray(y, dtype=int).ravel()
        nombres = [str(nombres_originales[i]) for i in ids_usados]

        # 4. Alto y ancho salen de images.shape; X aplana una foto por fila.
        alto, ancho = int(images.shape[1]), int(images.shape[2])
        X = images.reshape(images.shape[0], -1).astype(float)

        # Seguridad: Olivetti trabaja en [0, 1]; LFW debe quedar igual.
        if X.max() > 1.5:
            X = X / 255.0
            images = images / 255.0

        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        # 5. Validar antes de devolver.
        validar_conjunto(conjunto)
        return conjunto