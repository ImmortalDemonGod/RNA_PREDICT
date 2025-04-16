#!/bin/bash

###############################################################################
# CONFIGURATION
###############################################################################
# Folders you want to process
folders=(RNA_NET SPOT_RNA_PDB_dataset bpRNA kaggle)

# Log file (unique name with timestamp). You can also hardcode a filename.
LOG_FILE="commit_individual_files_$(date +'%Y%m%d_%H%M%S').log"

###############################################################################
# LOGGING SETUP
###############################################################################
# We’ll redirect all script output (stdout + stderr) through tee,
# appending to LOG_FILE. This way you see everything on the console
# AND it gets written to the log file.

exec > >(tee -a "$LOG_FILE") 2>&1

# Alternatively, you could do something like:
# exec &> >(ts "[%Y-%m-%d %H:%M:%S]" | tee -a "$LOG_FILE")
# ...which timestamps each line automatically (using the 'ts' command from 'moreutils').

# Simple manual logging functions:
log_info() {
  # Prints timestamp + [INFO] + message
  echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO]  $*"
}

log_error() {
  # Prints timestamp + [ERROR] + message
  echo "$(date +'%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2
}

###############################################################################
# PROGRESS BAR FUNCTION
###############################################################################
print_progress() {
  local current=$1
  local total=$2
  local width=50  # bar width in characters

  local percent=$(( current * 100 / total ))
  local num_hash=$(( percent * width / 100 ))

  local bar
  bar=$(printf '%0.s#' $(seq 1 $num_hash))
  local spaces=$(( width - num_hash ))
  local space_str
  space_str=$(printf '%0.s ' $(seq 1 $spaces))

  echo -ne "[$bar$space_str] $percent%% ($current/$total)\r"
}

###############################################################################
# MAIN SCRIPT
###############################################################################
log_info "===== Starting commit_individual_files.sh script ====="
log_info "Log file: $LOG_FILE"

for folder in "${folders[@]}"; do
  log_info "Processing folder: $folder"

  #########################
  # STEP 1: RENAME FILES WITH SPACES
  #########################
  # For each file that has spaces in its name, rename by replacing spaces with underscores.
  while IFS= read -r file; do
    new_file="$(echo "$file" | sed 's/ /_/g')"
    if [[ "$file" != "$new_file" ]]; then
      log_info "Renaming \"$file\" -> \"$new_file\""
      new_dir="$(dirname "$new_file")"
      mkdir -p "$new_dir"
      if mv "$file" "$new_file"; then
        log_info "Successfully renamed."
      else
        log_error "Failed to rename \"$file\" -> \"$new_file\""
      fi
    fi
  done < <(
    find "$folder" -type f -name "* *" -not -path '*/.git/*'
  )

  #########################
  # STEP 2: GATHER FILES AND SORT BY SIZE (SMALLEST FIRST)
  #########################
  # On macOS: stat -f "%z %N"
  # On Linux: stat -c "%s %n"
  file_list=$(
    find "$folder" -type f -not -path '*/.git/*' -not -name '.DS_Store' \
      -exec stat -f "%z %N" {} + 2>/dev/null \
    | sort -n
  )

  if [[ -z "$file_list" ]]; then
    log_info "No files found in $folder (or all already committed)."
    continue
  fi

  # Build an array of just file paths, ignoring the size column:
  files=()
  while IFS= read -r line; do
    # Each line is "SIZE PATH"
    path=$(echo "$line" | cut -d ' ' -f2-)
    files+=("$path")
  done <<< "$file_list"

  total=${#files[@]}
  log_info "Found $total files in $folder."

  #########################
  # STEP 3: COMMIT EACH FILE
  #########################
  local_count=0
  for file in "${files[@]}"; do
    ((local_count++))
    git add "$file" >/dev/null 2>&1
    if git commit -m "Add $(basename "$file")" >/dev/null 2>&1; then
      print_progress "$local_count" "$total"
    else
      # If something fails, log an error
      log_error "Failed to commit: $file"
    fi
  done

  # Move to a new line after finishing progress for this folder
  echo
  log_info "Done processing $folder."
done

#########################
# STEP 4: PUSH
#########################
if git push origin main; then
  log_info "Push to origin main successful."
else
  log_error "Push to origin main failed."
fi

log_info "===== Script completed. ====="