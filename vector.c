#include <stdio.h>
#include <stdlib.h>

void vector_init(vector *);
int vector_size(vector *);
static void vector_resize(vector *, int);
void vector_add(vector *, void *);
void vector_set(vector *, int, void *);
void *vector_get(vector *, int);
void vector_delete(vector *, int);
void vector_free(vector *);
void *vector_front(vector *);
void *vector_back(vector *);
bool vector_empty(vector *);

typedef struct vector
{
   void **items;
   int capacity;
   int size;
} vector;

void vector_init(vector *v)
{
   if(!v)
      v = malloc(sizeof(vector));
   v->capacity = 0;
   v->size = 0;
}

void vector_init_n(vector *v, int capacity)
{
   if(!v)
      v = malloc(sizeof(vector));
   v->size = 0;
   v->capacity = capacity;
   v->items = malloc(sizeof(void*) * capacity);
}

int vector_size(vector *v)
{
   if(!v)
      vector_init(v);
   return v->size;
}

static void vector_resize(vector *v, int capacity)
{
   void **items;
   if(capacity == 0)
   {
      capacity = 1;
      items = malloc(sizeof(void *) * 1);
   }
   else
      items = realloc(v->items,sizeof(void *) * capacity);
   if(items)
   {
      v->items = items;
      v->capacity = capacity;
   }
}

void vector_add(vector *v, void *item)
{
   if(!v)
      vector_init_n(v, 1);

   if(v->capacity == 0)
   {
      v->items = malloc(sizeof(void*));
      v->capacity = 1;
   }

   if(v->size == v->capacity)
      vector_resize(v, v->capacity * 2);

   v->items[v->size++] = item;
}

void vector_addfront(vector *v, void *item)
{
   if(!v)
      vector_init(v);

   if(v->size == v->capacity)
      vector_resize(v, v->capacity * 2);

   v->size++;
   for(int i = 1; i < v->size; i++)
      v->items[i] = v->items[i-1];
   v->items[0] = item;
}

void vector_set(vector *v, int index, void *item) 
{
   if(!v)
      vector_init_1(v);
   if(index >= 0 && index < v->size)
      v->items[index] = item;
}

void *vector_get(vector *v, int index)
{
   if(index >= 0 && index < v->size)
      return v->items[index];
   return NULL;
}

void vector_delete(vector *v, int index)
{

   if(index < 0 || index >= v->size)
      return;

   for(int i = index; i < v->size-1; i++)
      v->items[i] = v->items[i+1];

   v->items[v->size-1] = NULL;
   v->size--;
   if(v->size > 0 && v->size == v->capacity/4)
      vector_resize(v, v->capacity/2);
}

void vector_free(vector* v)
{
   free(v->items);
   v->size = 0;
   v->capacity = 0;
   free(v);
}

void *vector_front(vector *v)
{
   if(!v->size)
      return NULL;
   return v->items[0];
}

void *vector_back(vector *v)
{
   if(!v->size)
      return NULL;
   return v->items[v->size-1];
}

int vector_empty(vector *v)
{
   if(!v)
      return 1;
   return (v->size == 0);
}

void vector_swap(vector* v, vector* u)
{
    vector temp;
    temp = *v;
    *v = *u;
    *u = temp;
}

void vector_assign(vector* v, vector* u)
{
   if(!v)
      vector_init(v);

   if(!u)
      vector_init(u);

   *u = *v;
}