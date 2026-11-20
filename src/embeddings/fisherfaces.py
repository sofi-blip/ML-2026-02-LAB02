"""Fisherfaces: PCA comprime y LDA discrimina."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    1. PCA(d_pca) se ajusta solo con X_train (ahí se estima μ).
    2. LDA(d_lda) se ajusta sobre la salida del PCA, usando y_train.
    3. TEST pasa por las dos transformaciones, sin reajustar.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        self._pca: ExtractorPCA | None = None
        self._lda: ExtractorLDA | None = None

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: LDA usa las identidades.")

        n_muestras = X.shape[0]
        n_clases = int(np.unique(y).size)

        if self.d_lda > n_clases - 1:
            raise ValueError(
                f"d_lda={self.d_lda} supera C-1={n_clases - 1}. Reduzca d_lda."
            )
        # Con más de N-C componentes, la dispersión intra-clase queda singular.
        if self.d_pca > n_muestras - n_clases:
            raise ValueError(
                f"d_pca={self.d_pca} supera N-C={n_muestras - n_clases}. "
                "Reduzca d_pca para que LDA sea estable."
            )

        # Paso 1: PCA solo con TRAIN (no usa y).
        self._pca = ExtractorPCA(n_components=self.d_pca)
        Z_pca = self._pca.fit_transform(X)

        # Paso 2: LDA sobre el embedding PCA de TRAIN, con y_train.
        self._lda = ExtractorLDA(n_components=self.d_lda)
        self._lda.fit(Z_pca, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._exigir_ajuste()
        # Paso 3: aplica ambas proyecciones SIN reajustar.
        Z = self._lda.transform(self._pca.transform(X))
        return self.comprobar(Z, X.shape[0])

    def componentes_imagen(self, alto: int, ancho: int) -> np.ndarray:
        """Cada fisherface vista como imagen H×W (para la figura del informe)."""
        self._exigir_ajuste()
        W_pca = self._pca._modelo.components_                    # (d_pca, p)
        W_lda = self._lda._modelo.scalings_[:, : self.d_lda]     # (d_pca, d_lda)
        pesos = W_lda.T @ W_pca                                  # (d_lda, p)
        if pesos.shape[1] != alto * ancho:
            raise ValueError("La fisherface no coincide con alto × ancho.")
        return pesos.reshape(-1, alto, ancho)

    def _exigir_ajuste(self) -> None:
        if self._pca is None or self._lda is None:
            raise RuntimeError("Fisherfaces: llame a fit antes de transform.")