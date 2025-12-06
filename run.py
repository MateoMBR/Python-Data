#!/usr/bin/env python3
"""
🚀 MAIN LAUNCHER - Exécute l'ensemble du pipeline d'analyse
Lance tous les modules : analysis.py, storytelling.py, attack_recommender.py
"""

import sys
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS = [
    {
        'name': 'Analysis.py',
        'path': 'src/analysis.py',
        'description': 'Exploration & Nettoyage & KPI'
    },
    {
        'name': 'Storytelling.py',
        'path': 'src/storytelling.py',
        'description': 'Datatelling & Recommandations'
    }
]

# ============================================================================
# LAUNCHER
# ============================================================================

def print_header():
    """Affiche le header"""
    print("\n" + "#"*70)
    print("# 🚀 PHISHING ANALYSIS PIPELINE - MAIN LAUNCHER")
    print("#"*70)
    print("\n")

def print_menu():
    """Affiche le menu"""
    print("📋 SÉLECTIONNEZ UN SCRIPT À EXÉCUTER :")
    print("─" * 70)
    for i, script in enumerate(SCRIPTS, 1):
        print(f"{i}. {script['name']:<30} - {script['description']}")
    print(f"{len(SCRIPTS)+1}. {'Tout exécuter':<30} - Lance tous les scripts")
    print(f"0. {'Quitter':<30} - Exit")
    print("─" * 70)

def run_script(script_path):
    """Exécute un script Python"""
    try:
        print(f"\n🔄 Exécution de {script_path}...")
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution : {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {script_path}")
        return False

def run_all_scripts():
    """Exécute tous les scripts"""
    print(f"\n🚀 LANCEMENT DE TOUS LES SCRIPTS")
    print("="*70)
    
    results = []
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] Exécution {script['name']}...")
        success = run_script(script['path'])
        results.append((script['name'], success))
        print("─"*70)
    
    # Résumé
    print(f"\n\n{'='*70}")
    print("📊 RÉSUMÉ EXÉCUTION")
    print("="*70)
    for script_name, success in results:
        status = "✅ SUCCÈS" if success else "❌ ERREUR"
        print(f"{status} : {script_name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nTotal : {successful}/{len(SCRIPTS)} scripts exécutés avec succès")
    print("\n")

def interactive_mode():
    """Mode interactif"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("👉 Votre choix (0-4) : ").strip()
            
            if choice == "0":
                print("👋 Au revoir !")
                break
            elif choice == str(len(SCRIPTS) + 1):
                run_all_scripts()
            elif choice in [str(i) for i in range(1, len(SCRIPTS) + 1)]:
                script_idx = int(choice) - 1
                script = SCRIPTS[script_idx]
                run_script(script['path'])
                input("\n👇 Appuyez sur Entrée pour continuer...")
            else:
                print("❌ Choix invalide. Veuillez réessayer.")
                input("👇 Appuyez sur Entrée pour continuer...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")
            input("👇 Appuyez sur Entrée pour continuer...")

def cli_mode(args):
    """Mode CLI (arguments)"""
    if not args or args[0] in ['-h', '--help', 'help']:
        print(f"""
Usage: python3 run.py [OPTION]

Options:
    all                 Exécuter tous les scripts
    analysis            Exécuter analysis.py
    storytelling        Exécuter storytelling.py
    attack_recommender  Exécuter attack_recommender.py
    -h, --help         Afficher cette aide

Examples:
    python3 run.py all
    python3 run.py analysis
        """)
        return
    
    command = args[0].lower()
    
    if command == 'all':
        run_all_scripts()
    elif command == 'analysis':
        run_script('src/analysis.py')
    elif command == 'storytelling':
        run_script('src/storytelling.py')
    elif command == 'attack_recommender':
        run_script('src/attack_recommender.py')
    else:
        print(f"❌ Commande inconnue : {command}")
        print("Utilisez 'python3 run.py -h' pour l'aide")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Mode CLI
        cli_mode(sys.argv[1:])
    else:
        # Mode interactif
        interactive_mode()
