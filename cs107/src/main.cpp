#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define NUM_STRINGS 5

const char* inputs[NUM_STRINGS] = {
    "bumfuzzled",
    "palaeonanthropology",
    "equitability",
    "sage",
    "antidisestablishmentarianism",
};

/*
 * If length <= 16, then the string is contained within chars
 * Otherwise, the first eight characters contain chars,
 *     and the rest of the bytes hold a pointer to the remaining chars
 */
struct string {
    size_t length;
    char chars[16];
};

/*
 * Converts a string literal to a string struct 
 */
struct string encode(const char* input) {
    struct string str;

    str.length = strlen(input);
    int numBytes = strlen(input) + 1;
    if (numBytes <= 16) {
        strcpy(str.chars, input);

    } else {
        strncpy(str.chars, input, 8);

        // Allocate new memory for overflow and fill it
        int overflowBytes = numBytes - 8;
        char* overflowBuffer = (char*) malloc(overflowBytes);
        strcpy(overflowBuffer, input + 8);

        // Write the pointer into the struct
        *((char**) (str.chars + 8)) = overflowBuffer;
    }

    return str;
}

/*
 * Decodes and prints a string struct to stdout
 */
void decodePrint(struct string* str) {
    if (str->length <= 16) {
        printf("%s\n", str->chars);
    } else {
        for (int i = 0; i < 8; ++i) {
            printf("%c", str->chars[i]);
        }
        printf("%s\n", *((char**)(str->chars + 8)));
    }
}

/*
 * Serializes an array of struct strings, returning
 *     a pointer to a string with the strings concatenated
 */
char* serialize(struct string strings[], size_t length) {
    char* serialization = strdup("");
    size_t str_len = 1; // includes null-terminated character;

    for (int i = 0; i < length; ++i) {
        size_t cur_len = strings[i].length;
        serialization = (char*) realloc(serialization, str_len + cur_len);
        char* pos = serialization + str_len - 1;

        if (cur_len <= 16) {
            strcpy(pos, strings[i].chars);
        } else {
            strncpy(pos, strings[i].chars, 8);
            char* remaining = *(char**)(strings[i].chars + 8);
            strcpy(pos + 8, remaining);
        }

        str_len += cur_len;
    }

    return serialization;
}

int main() {
    // Create struct string objects
    struct string strs[NUM_STRINGS];
    for (int i = 0; i < NUM_STRINGS; ++i) {
        strs[i] = encode(inputs[i]);
    }

    // Run Final Test
    const char* serialization = serialize(strs, NUM_STRINGS);
    printf("Serialization:\n%s\n", serialization);
}
