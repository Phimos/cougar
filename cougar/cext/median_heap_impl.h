#ifdef T

#include "math.h"
#include "stdlib.h"

#define concat(a, b) a##_##b

#define median_heap_node(dtype) struct concat(median_heap_node, dtype)
#define median_heap(dtype) struct concat(median_heap, dtype)
#define method(name, dtype) concat(median_heap_##name, dtype)

median_heap_node(T) {
    char is_min;
    T value;
    size_t index;
};

median_heap(T) {
    median_heap_node(T) * buffer;
    median_heap_node(T) * buffer_end;

    median_heap_node(T) * *heap;
    median_heap_node(T) * *min_heap;
    median_heap_node(T) * *max_heap;

    size_t min_size, min_capacity;
    size_t max_size, max_capacity;

    median_heap_node(T) * head;
    median_heap_node(T) * tail;
};

static inline void method(reset, T)(median_heap(T) * heap) {
    heap->min_size = heap->max_size = 0;
    heap->head = heap->tail = NULL;
}

static inline median_heap(T) * method(init, T)(size_t min_heap_size, size_t max_heap_size) {
    const size_t size = min_heap_size + max_heap_size;
    median_heap(T)* heap = (median_heap(T)*)malloc(sizeof(median_heap(T)));
    heap->buffer = (median_heap_node(T)*)malloc(sizeof(median_heap_node(T)) * size);
    heap->heap = (median_heap_node(T)**)malloc(sizeof(median_heap_node(T)*) * size);
    heap->buffer_end = heap->buffer + size - 1;

    heap->min_heap = heap->heap;
    heap->max_heap = heap->heap + min_heap_size;

    heap->min_capacity = min_heap_size;
    heap->max_capacity = max_heap_size;

    method(reset, T)(heap);
    return heap;
}

static inline void method(free, T)(median_heap(T) * heap) {
    free(heap->heap);
    free(heap->buffer);
    free(heap);
}

static inline size_t method(get_parent, T)(size_t index) {
    return (index - 1) >> 1;
}

static inline size_t method(get_left_child, T)(size_t index) {
    return (index << 1) + 1;
}

static inline size_t method(get_right_child, T)(size_t index) {
    return (index << 1) + 2;
}

static inline size_t method(get_smallest_child, T)(median_heap_node(T) * *heap, size_t index, size_t size) {
    const size_t left = method(get_left_child, T)(index);
    const size_t right = method(get_right_child, T)(index);
    if (right < size) {
        return heap[left]->value < heap[right]->value ? left : right;
    } else if (left < size) {
        return left;
    } else {
        return size;
    }
}

static inline size_t method(get_largest_child, T)(median_heap_node(T) * *heap, size_t index, size_t size) {
    const size_t left = method(get_left_child, T)(index);
    const size_t right = method(get_right_child, T)(index);
    if (right < size) {
        return heap[left]->value > heap[right]->value ? left : right;
    } else if (left < size) {
        return left;
    } else {
        return size;
    }
}

static inline void method(swap, T)(median_heap_node(T) * *heap, size_t i, size_t j) {
    median_heap_node(T)* temp = heap[i];
    heap[i] = heap[j];
    heap[j] = temp;

    heap[i]->index = i;
    heap[j]->index = j;
}

static inline void method(max_sift_up, T)(median_heap_node(T) * *heap, size_t index) {
    median_heap_node(T)* node = heap[index];
    size_t parent = method(get_parent, T)(index);
    while (index > 0 && node->value > heap[parent]->value) {
        method(swap, T)(heap, index, parent);
        index = parent;
        parent = method(get_parent, T)(index);
    }
    heap[index] = node;
}

static inline void method(min_sift_up, T)(median_heap_node(T) * *heap, size_t index) {
    median_heap_node(T)* node = heap[index];
    size_t parent = method(get_parent, T)(index);
    while (index > 0 && node->value < heap[parent]->value) {
        method(swap, T)(heap, index, parent);
        index = parent;
        parent = method(get_parent, T)(index);
    }
    heap[index] = node;
}

static inline void method(max_sift_down, T)(median_heap_node(T) * *heap, size_t index, size_t size) {
    median_heap_node(T)* node = heap[index];
    size_t child = method(get_largest_child, T)(heap, index, size);
    while (child < size && node->value < heap[child]->value) {
        method(swap, T)(heap, index, child);
        index = child;
        child = method(get_largest_child, T)(heap, index, size);
    }
    heap[index] = node;
}

static inline void method(min_sift_down, T)(median_heap_node(T) * *heap, size_t index, size_t size) {
    median_heap_node(T)* node = heap[index];
    size_t child = method(get_smallest_child, T)(heap, index, size);
    while (child < size && node->value > heap[child]->value) {
        method(swap, T)(heap, index, child);
        index = child;
        child = method(get_smallest_child, T)(heap, index, size);
    }
    heap[index] = node;
}

static inline void method(max_insert, T)(median_heap(T) * heap, median_heap_node(T) * node) {
    node->is_min = 0;
    heap->max_heap[heap->max_size] = node;
    node->index = heap->max_size;
    method(max_sift_up, T)(heap->max_heap, heap->max_size);
    heap->max_size++;
}

static inline void method(min_insert, T)(median_heap(T) * heap, median_heap_node(T) * node) {
    node->is_min = 1;
    heap->min_heap[heap->min_size] = node;
    node->index = heap->min_size;
    method(min_sift_up, T)(heap->min_heap, heap->min_size);
    heap->min_size++;
}

static inline void method(max_remove, T)(median_heap(T) * heap, median_heap_node(T) * node) {
    const size_t index = node->index;
    heap->max_size--;
    method(swap, T)(heap->max_heap, index, heap->max_size);
    method(max_sift_up, T)(heap->max_heap, index);
    method(max_sift_down, T)(heap->max_heap, index, heap->max_size);
}

static inline void method(min_remove, T)(median_heap(T) * heap, median_heap_node(T) * node) {
    const size_t index = node->index;
    heap->min_size--;
    method(swap, T)(heap->min_heap, index, heap->min_size);
    method(min_sift_up, T)(heap->min_heap, index);
    method(min_sift_down, T)(heap->min_heap, index, heap->min_size);
}

static inline void method(push, T)(median_heap(T) * heap, T value) {
    if (heap->head == NULL) {
        heap->head = heap->tail = heap->buffer;
    } else {
        if (heap->tail == heap->buffer_end) {
            heap->tail = heap->buffer;
        } else {
            ++(heap->tail);
        }
    }
    heap->tail->value = value;

    if (heap->min_size <= heap->max_size) {
        method(min_insert, T)(heap, heap->tail);
    } else {
        method(max_insert, T)(heap, heap->tail);
    }

    if (heap->min_size == 0 || heap->max_size == 0) {
        return;
    }

    median_heap_node(T)* min_top = heap->min_heap[0];
    median_heap_node(T)* max_top = heap->max_heap[0];

    if (min_top->value < max_top->value) {
        heap->min_heap[0] = max_top;
        heap->max_heap[0] = min_top;
        max_top->is_min = 1;
        min_top->is_min = 0;
        method(min_sift_down, T)(heap->min_heap, 0, heap->min_size);
        method(max_sift_down, T)(heap->max_heap, 0, heap->max_size);
    }
}

static inline void method(pop, T)(median_heap(T) * heap) {
    if (heap->min_size == 0 && heap->max_size == 0) {
        return;
    }

    median_heap_node(T)* node = heap->head;

    if (heap->head == heap->tail) {
        heap->head = heap->tail = NULL;
    } else {
        if (heap->head == heap->buffer_end) {
            heap->head = heap->buffer;
        } else {
            ++(heap->head);
        }
    }

    if (node->is_min) {
        method(min_remove, T)(heap, node);
    } else {
        method(max_remove, T)(heap, node);
    }

    if (heap->min_size < heap->max_size) {
        median_heap_node(T)* max_top = heap->max_heap[0];
        method(max_remove, T)(heap, max_top);
        method(min_insert, T)(heap, max_top);
    } else if (heap->min_size > heap->max_size + 1) {
        median_heap_node(T)* min_top = heap->min_heap[0];
        method(min_remove, T)(heap, min_top);
        method(max_insert, T)(heap, min_top);
    }
}

static inline double method(query_median, T)(median_heap(T) * heap) {
    if (heap->min_size == heap->max_size) {
        return (heap->min_heap[0]->value + heap->max_heap[0]->value) * 0.5;
    } else {
        return heap->min_heap[0]->value;
    }
}

#undef method
#undef median_heap
#undef median_heap_node
#undef concat

#endif