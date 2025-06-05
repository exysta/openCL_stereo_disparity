CC = g++
CFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -framework OpenCL

# Définir les répertoires
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Créer les répertoires de sortie s'ils n'existent pas
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

# Définir les fichiers source et objets
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Définir l'exécutable
EXECUTABLE = $(BIN_DIR)/stereo_disparity

# Règle par défaut
all: $(EXECUTABLE)

# Règle pour l'exécutable
$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Règle pour les fichiers objets
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Règle pour nettoyer
clean:
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

# Règle pour exécuter
run: $(EXECUTABLE)
	$(EXECUTABLE)

.PHONY: all clean run
