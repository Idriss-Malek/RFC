# Compilateur C++
CXX = g++

# Options de compilation
CXXFLAGS = -Wall -Wextra -std=c++17

# Fichiers sources
SRC = $(wildcard src/*.cpp)

# Fichiers objets
OBJ = $(SRC:.cpp=.o)

# Dossier de destination pour les fichiers .o
BUILD = build

# Fichier exécutable
EXE = MonProjet

# Règles de compilation
$(EXE) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/$@ $^

# Règles de construction
.PHONY : all clean

all : $(EXE)

clean :
	rm -f $(OBJ) $(EXE)