# Image du frontend React : construction avec Node, puis service avec nginx.
#
# Construction (depuis la racine du dépôt) :
#   docker build -t fissures-frontend frontend
# En général on passe par docker-compose.yml, qui lance aussi le backend.

# --- Étape 1 : construire le site (fichiers statiques dans dist/) ---
FROM node:20-alpine AS build
WORKDIR /app

# Les dépendances d'abord : cette couche reste en cache tant que package*.json ne change pas.
COPY package.json package-lock.json ./
RUN npm install

COPY . .
# Adresse de l'API vue par le navigateur. "/api" : nginx la renvoie vers le backend.
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# --- Étape 2 : servir le site avec nginx (image finale légère, sans Node) ---
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
