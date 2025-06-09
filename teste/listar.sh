#!/bin/bash

# Script de administração do PostgreSQL com funcionalidades avançadas
# Uso: ./pg_admin.sh

# Carrega variáveis do .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo -e "[ERROR] Arquivo .env não encontrado. Execute o script na pasta raiz do projeto."
    exit 1
fi

# Configurações do PostgreSQL
PG_HOST=${PG_HOST:-localhost}
PG_PORT=${PG_PORT:-5432}
PG_DB=${PG_DB:-vector}
PG_USER=${PG_USER:-postgres}  # Deve ser um superusuário para gerenciar usuários
PG_PASSWORD=${PG_PASSWORD:-postgres}
PG_SCHEMA=${PG_SCHEMA:-public}

# Função: Pausar execução
pause() {
    read -p $'\nPressione Enter para continuar...' -r
}

# Função: Listar bancos de dados
list_databases() {
    echo -e "\n[INFO] Listando bancos de dados..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        --quiet \
        --tuples-only \
        --command "\\l"
    pause
}

# Função: Listar tabelas no banco atual
list_tables() {
    echo -e "\n[INFO] Listando tabelas no banco '$PG_DB' (schema: $PG_SCHEMA)..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "$PG_DB" \
        --quiet \
        --tuples-only \
        --command "SELECT table_name FROM information_schema.tables WHERE table_schema = '$PG_SCHEMA';"
    pause
}

# Função: Listar registros da tabela 'documents'
list_records() {
    echo -e "\n[INFO] Exibindo registros da tabela '${PG_SCHEMA}.documents'..."
    
    read -p "[INPUT] Quantos registros mostrar? (Digite 'all' para todos, ou um número): " limit_input
    limit_input=$(echo "$limit_input" | tr '[:lower:]' '[:upper:]')
    
    if [[ "$limit_input" == "ALL" ]]; then
        limit_clause=""
    elif [[ "$limit_input" =~ ^[0-9]+$ ]]; then
        limit_clause="LIMIT $limit_input"
    else
        echo -e "[ERROR] Entrada inválida. Mostrando os 5 primeiros registros."
        limit_clause="LIMIT 5"
    fi

    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "$PG_DB" \
        -x \
        -t \
        --quiet \
        --command "
            SELECT id, metadata->>'filename' AS filename, LENGTH(content) AS content_length, metadata 
            FROM ${PG_SCHEMA}.documents
            $limit_clause;
        "
    pause
}

# Função: Limpar registros da tabela 'documents'
clean_tables() {
    echo -e "\n[INFO] Tabela alvo: ${PG_SCHEMA}.documents"
    read -p "[CONFIRM] Tem certeza que deseja limpar a tabela? (s/N): " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Limpando tabela '${PG_SCHEMA}.documents'..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "$PG_DB" \
        --quiet \
        --command "TRUNCATE TABLE ${PG_SCHEMA}.documents RESTART IDENTITY;"
    echo "[SUCCESS] Tabela '${PG_SCHEMA}.documents' limpa com sucesso."
    pause
}

# Função: Excluir uma tabela específica
drop_table() {
    echo -e "\n[INFO] Excluir uma tabela do banco '$PG_DB' (schema: $PG_SCHEMA)..."
    read -p "[INPUT] Nome da tabela a ser excluída: " table_name
    read -p "[CONFIRM] Tem certeza que deseja excluir a tabela '${PG_SCHEMA}.$table_name'? (s/N): " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Excluindo tabela '${PG_SCHEMA}.$table_name'..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "$PG_DB" \
        --quiet \
        --command "DROP TABLE IF EXISTS ${PG_SCHEMA}.$table_name CASCADE;"
    echo "[SUCCESS] Tabela '${PG_SCHEMA}.$table_name' excluída com sucesso."
    pause
}

# Função: Excluir um banco de dados
drop_database() {
    echo -e "\n[INFO] Excluir um banco de dados..."
    read -p "[INPUT] Nome do banco de dados a ser excluído: " db_name
    read -p "[CONFIRM] Tem certeza que deseja excluir o banco '$db_name'? (s/N): " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Excluindo banco de dados '$db_name'..."
    PGPASSWORD="$PG_PASSWORD" dropdb \
        -h "$PG_HOST" \
        -p "$PG_PORT" \
        -U "$PG_USER" \
        "$db_name"
    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Banco de dados '$db_name' excluído com sucesso."
    else
        echo "[ERROR] Falha ao excluir o banco de dados '$db_name'."
    fi
    pause
}

# Função: Listar usuários
list_users() {
    echo -e "\n[INFO] Listando usuários do PostgreSQL..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        --quiet \
        --tuples-only \
        --command "\\du"
    pause
}

# Função: Criar usuário
create_user() {
    echo -e "\n[INFO] Criando novo usuário..."
    read -p "[INPUT] Nome do novo usuário: " new_user
    read -p "[INPUT] Senha do novo usuário: " -s new_password
    echo

    if [[ -z "$new_user" ]]; then
        echo -e "[ERROR] Nome de usuário inválido. Operação cancelada."
        pause
        return
    fi

    echo -e "[CONFIRM] Criar usuário '$new_user'? (s/N)"
    read -p "> " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    # Conecta-se ao banco padrão 'postgres'
    echo -e "[INFO] Criando usuário '$new_user' no banco 'postgres'..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "postgres" \
        --quiet \
    --command "CREATE USER $new_user WITH PASSWORD '$new_password';"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Usuário '$new_user' criado com sucesso."
    else
        echo "[ERROR] Falha ao criar usuário '$new_user'."
    fi
    pause
}

# Função: Excluir usuário
delete_user() {
    echo -e "\n[INFO] Excluindo usuário..."
    read -p "[INPUT] Nome do usuário a ser excluído: " user_to_delete
    read -p "[CONFIRM] Tem certeza que deseja excluir o usuário '$user_to_delete'? (s/N): " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Excluindo usuário '$user_to_delete' no banco 'postgres'..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "postgres" \
        --quiet \
    --command "DROP USER IF EXISTS $user_to_delete;"

    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Usuário '$user_to_delete' excluído com sucesso."
    else
        echo "[ERROR] Falha ao excluir usuário '$user_to_delete'."
    fi
    pause
}

# Função: Criar banco de dados
create_database() {
    echo -e "\n[INFO] Criando novo banco de dados..."
    read -p "[INPUT] Nome do novo banco de dados: " new_db
    read -p "[CONFIRM] Tem certeza que deseja criar o banco '$new_db'? (s/N): " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Criando banco de dados '$new_db'..."
    PGPASSWORD="$PG_PASSWORD" createdb \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        "$new_db"
    
    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Banco de dados '$new_db' criado com sucesso."
    else
        echo "[ERROR] Falha ao criar banco de dados '$new_db'."
    fi
    pause
}

# Função: Dar permissão total a um usuário em um banco específico
grant_full_access() {
    echo -e "\n[INFO] Concedendo permissões totais..."
    read -p "[INPUT] Nome do usuário: " target_user
    read -p "[INPUT] Nome do banco de dados: " target_db

    # Confirmação
    echo -e "[CONFIRM] Conceder permissões totais ao usuário '$target_user' no banco '$target_db'? (s/N)"
    read -p "> " confirm
    if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
        echo "[INFO] Operação cancelada."
        pause
        return
    fi

    echo -e "[INFO] Concedendo permissões ao usuário '$target_user' no banco '$target_db'..."
    PGPASSWORD="$PG_PASSWORD" psql \
        -h "$PG_HOST" \
        -U "$PG_USER" \
        -p "$PG_PORT" \
        -d "postgres" \
        --quiet \
    --command "GRANT ALL PRIVILEGES ON DATABASE $target_db TO $target_user;"
    
    if [[ $? -eq 0 ]]; then
        echo "[SUCCESS] Permissões concedidas com sucesso."
    else
        echo "[ERROR] Falha ao conceder permissões ao usuário '$target_user'."
    fi
    pause
}

# Submenu: Listar Informações
listar_submenu() {
    while true; do
        clear
        echo -e "*** Submenu: Listar ***"
        echo "1 - Listar Bancos de Dados"
        echo "2 - Listar Tabelas"
        echo "3 - Listar Registros da Tabela 'documents'"
        echo "0 - Voltar"
        read -p "> " subchoice

        case $subchoice in
            1) list_databases ;;
            2) list_tables ;;
            3) list_records ;;
            0) break ;;
            *) echo -e "[ERROR] Opção inválida. Tente novamente.\n" ;;
        esac
    done
}

# Submenu: Excluir Dados
excluir_submenu() {
    while true; do
        clear
        echo -e "*** Submenu: Excluir ***"
        echo "1 - Excluir Banco de Dados"
        echo "2 - Excluir Tabela"
        echo "0 - Voltar"
        read -p "> " subchoice

        case $subchoice in
            1) drop_database ;;
            2) drop_table ;;
            0) break ;;
            *) echo -e "[ERROR] Opção inválida. Tente novamente.\n" ;;
        esac
    done
}

# Submenu: Gerenciar Usuários
usuario_submenu() {
    while true; do
        clear
        echo -e "*** Submenu: Gerenciar Usuários ***"
        echo "1 - Listar Usuários"
        echo "2 - Criar Usuário"
        echo "3 - Excluir Usuário"
        echo "4 - Criar Banco de Dados"
        echo "5 - Dar Permissão Total"
        echo "0 - Voltar"
        read -p "> " subchoice

        case $subchoice in
            1) list_users ;;
            2) create_user ;;
            3) delete_user ;;
            4) create_database ;;
            5) grant_full_access ;;
            0) break ;;
            *) echo -e "[ERROR] Opção inválida. Tente novamente.\n" ;;
        esac
    done
}

# Menu Principal
main_menu() {
    while true; do
        clear
        echo -e "*** Admin PostgreSQL ***"
        echo "1 - Listar Informações"
        echo "2 - Excluir Dados"
        echo "3 - Gerenciar Usuários e Bancos"
        echo "0 - Sair"
        read -p "> " choice

        case $choice in
            1) listar_submenu ;;
            2) excluir_submenu ;;
            3) usuario_submenu ;;
            0) 
                echo -e "[INFO] Saindo..."
                exit 0 ;;
            *)
                echo -e "[ERROR] Opção inválida. Tente novamente.\n" ;;
        esac
    done
}

# Execução do script
main_menu