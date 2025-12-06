#!/bin/bash

# 📊 Setup Script pour analyze-python
# Installe l'environnement et les dépendances

set -e

echo "🚀 Configuration du projet analyze-python"
echo "=========================================="

# Créer l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
else
    echo "✓ Environnement virtuel existant"
fi

# Activer l'environnement
echo "🔌 Activation de l'environnement..."
source venv/bin/activate

# Installer les dépendances
echo "📚 Installation des dépendances..."
pip install -q -r requirements.txt

echo "✅ Setup complété !"
echo ""
echo "Pour exécuter l'analyse :"
echo "  source venv/bin/activate"
echo "  python3 src/analysis.py"
echo ""
