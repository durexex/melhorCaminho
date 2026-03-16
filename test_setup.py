"""
Script de verificação de setup para a integração LLM.
Valida que todas as dependências necessárias estão instaladas
e que as variáveis de ambiente estão configuradas.
"""
import sys


def check_imports() -> bool:
    """Verifica se todas as bibliotecas necessárias podem ser importadas."""
    errors: list[str] = []

    try:
        import langchain
        print(f"[OK] langchain {langchain.__version__}")
    except ImportError as e:
        errors.append(f"[FALHA] langchain: {e}")

    try:
        import langchain_core
        print(f"[OK] langchain-core {langchain_core.__version__}")
    except ImportError as e:
        errors.append(f"[FALHA] langchain-core: {e}")

    try:
        from langchain_openai import ChatOpenAI
        print("[OK] langchain-openai (ChatOpenAI)")
    except ImportError as e:
        errors.append(f"[FALHA] langchain-openai: {e}")

    try:
        from fpdf import FPDF
        print("[OK] fpdf2 (FPDF)")
    except ImportError as e:
        errors.append(f"[FALHA] fpdf2: {e}")

    try:
        from dotenv import load_dotenv
        print("[OK] python-dotenv (load_dotenv)")
    except ImportError as e:
        errors.append(f"[FALHA] python-dotenv: {e}")

    if errors:
        print("\n--- Erros encontrados ---")
        for err in errors:
            print(err)
        return False

    return True


def check_env_vars() -> bool:
    """Verifica se as variáveis de ambiente do LLM estão definidas no .env."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "")

    print(f"\nOPENAI_API_KEY: {'configurada' if openai_key and openai_key != 'sk-your-api-key-here' else 'NÃO configurada (placeholder)'}")
    print(f"LLM_MODEL: {llm_model or 'não definido (usará padrão gpt-4)'}")

    return True


def main() -> None:
    """Executa todas as verificações de setup."""
    print("=" * 50)
    print("Verificação de Setup - Integração LLM")
    print("=" * 50)

    print("\n--- Verificando importações ---")
    imports_ok = check_imports()

    print("\n--- Verificando variáveis de ambiente ---")
    env_ok = check_env_vars()

    print("\n" + "=" * 50)
    if imports_ok and env_ok:
        print("RESULTADO: Todas as verificações passaram!")
        print("O ambiente está pronto para a integração LLM.")
    else:
        print("RESULTADO: Algumas verificações falharam.")
        print("Revise os erros acima e instale as dependências faltantes.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
