import os
import subprocess
import sys

# --- Configuration ---
COMMIT_MESSAGE = "Actualizar base de conocimiento y DB"
# --- End Configuration ---

def run_command(command):
    """Runs a command and lets it print directly to the console."""
    print(f"--- Running: {' '.join(command)} ---")
    # Let the subprocess inherit stdout/stderr to avoid encoding issues in this script.
    # The child process's output will be streamed directly to the console.
    result = subprocess.run(command, shell=True, check=False)
    
    if result.returncode != 0:
        print(f"\n--- Command failed with exit code {result.returncode} ---")
    
    return result.returncode

def main():
    """Main function to run the update process."""
    print(">>> Paso 1: Actualizando la base de datos de ChromaDB con ingestar.py...")
    if run_command(["python", "ingestar.py"]) != 0:
        print("\nError al ingestar documentos. Abortando el script.")
        sys.exit(1)

    print("\n>>> Paso 2: Añadiendo archivos al área de preparación de Git...")
    run_command(["git", "add", "documentos_srt/"])

    print("\n>>> Paso 3: Creando el commit...")
    # Check if there are staged changes before committing
    status_result_after_add = subprocess.run(["git", "diff", "--staged", "--quiet"], shell=True)
    if status_result_after_add.returncode == 0:
        print("\nNo hay nuevos documentos o cambios en la base de datos para registrar en el repositorio.")
        print("El repositorio ya está actualizado.")
        sys.exit(0)

    commit_command = ["git", "commit", "-m", f'"{COMMIT_MESSAGE}"']
    if run_command(commit_command) != 0:
        print("\nError al crear el commit. Revisa los mensajes de error de Git. Abortando.")
        sys.exit(1)

    print("\n>>> Paso 4: Subiendo los cambios al repositorio remoto...")
    if run_command(["git", "push"]) != 0:
        print("\nError al subir los cambios (push). Revisa los mensajes de error de Git.")
        sys.exit(1)

    print("\n\nProceso completado. La base de conocimientos y el repositorio han sido actualizados.")

if __name__ == "__main__":
    main()