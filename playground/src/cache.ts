// Clés du Cache API, partagées par le worker d'inférence (écriture, nettoyage) et la page (déjà téléchargé ?).
export const CACHE_PREFIX = "regex-playground/";

/** Un cache par dossier d'export et par version (révision du Hub, ou playground_export_id en local). */
export const cacheKey = (files: string, version: string) => `${CACHE_PREFIX}${files}/${version}`;

/** Dossier d'export d'une clé, null si la clé n'est pas au playground. */
export const cacheFiles = (key: string) => (key.startsWith(CACHE_PREFIX) ? key.slice(CACHE_PREFIX.length).split("/")[0] : null);
