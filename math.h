#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void vector_init(vector *v);
void vector_init_n(vector* v, int capacity);
int vector_size(vector *v);
static void vector_resize(vector *v, int capacity);
void vector_add(vector *v, void *item);
void vector_set(vector *v, int idx, void *item);
void *vector_get(vector *v, int idx);
void vector_delete(vector *v, int idx);
void vector_free(vector *v);
void *vector_front(vector *v);
void *vector_back(vector *v);
bool vector_empty(vector *v);

typedef struct Scalar
{
    double val[4];
} Scalar ;

Scalar init_Scalar(double v1, double v2, double v3, double v4)
{
    Scalar s;
    s.val[0] = v1;
    s.val[1] = v2;
    s.val[2] = v3;
    s.val[3] = v4;
    return s;
}

typedef struct Point
{
    int x;
    int y;
} Point ;

typedef struct Vec3i
{
    int val[3];
} Vec3i ;

Vec3i vec3i(int a, int b, int c)
{
    Vec3i v;
    v.val[0] = a;
    v.val[1] = b;
    v.val[2] = c;
    return v;
}

typedef struct floatPoint
{
    float x;
    float y;
} floatPoint ;


typedef struct Rect
{
	int x;					  //!< x coordinate of the top-left corner
	int y;					  //!< y coordinate of the top-left corner 
	int width;				  //!< width of the rectangle
	int height;				  //!< height of the rectangle
} Rect ;

Rect init_Rect(int _x, int _y, int _width, int _height)
{
    Rect r;
	r->x = _x;
	r->y = _y;
	r->width = _width;
	r->height = _height;
    return r;
}

Rect createRect(Point p, Point q)
{
    Rect r;
    r.x = min(p.x, q.x);
    r.y = min(p.y, q.y);
    r.width = max(p.x, q.x) - r.x;
    r.height = max(p.y, q.y) - r.y;
    return r;
}

Rect performOR(Rect* a, Rect* b)
{
    if(a->width <= 0 || a->height <= 0)
        a = b;

    else if(b->width > 0 && b->height > 0)
    {
        int x1 = min(a->x, b->x);
        int y1 = min(a->y, b->y);
        a->width = max(a->x + a->width, b->x + b->width) - x1;
        a->height = max(a->y + a->height, b->y + b->height) - y1;
        a->x = x1;
        a->y = y1;
    }
    return *a;
}

bool equalRects(Rect a, Rect b)
{
    return (a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height);
}

// Represents a pair of ER's
typedef struct region_pair
{
    Point a;
    Point b;
} region_pair ;

region_pair init_region_pair(Point a, Point b)
{
    region_pair pair;
    pair.a = a;
    pair.b = b;
    return pair;
}

bool equalRegionPairs(region_pair r1, region_pair r2)
{
    return r1.a.x == r2.a.x && r1.a.y == r2.a.y && r1.b.x == r2.b.x && r1.b.y == r2.b.y;
}

// struct region_triplet
// Represents a triplet of ER's
typedef struct region_triplet
{
    Point a;
    Point b;
    Point c;
    line_estimates estimates;
} region_triplet ;

region_triplet init_region_triplet(Point _a, Point _b, Point _c)
{
    region_triplet triplet;
    triplet.a = _a;
    triplet.b = _b;
    triplet.c = _c;
    return triplet;
}

// struct region_sequence
// Represents a sequence of more than three ER's
typedef struct region_sequence
{
    vector* triplets; //region_triplet
};

region_sequence Region_sequence(region_triplet* t)
{
    region_sequence sequence;
    sequence.triplets = malloc(sizeof(vector));
    vector_add(sequence.triplets, t);
    return sequence;
}

// struct line_estimates
// Represents a line estimate (as above) for an ER's group
// i.e.: slope and intercept of 2 top and 2 bottom lines
typedef struct line_estimates
{
    float top1_a0;
    float top1_a1;
    float top2_a0;
    float top2_a1;
    float bottom1_a0;
    float bottom1_a1;
    float bottom2_a0;
    float bottom2_a1;
    int x_min;
    int x_max;
    int h_max;
} line_estimates ;


void swap(void *v1, void* v2)
{
    int temp = *(int *)v1;
    *(int *)v1 = *(int *)v2;
    *(int *)v2 = temp;
}

void sort_3ints(vector *v)
{
    if (*(int *)vector_get(v, 0) > *(int *)vector_get(v, 1))
        swap(vector_get(v, 0), vector_get(v, 1));

    if (*(int *)vector_get(v, 1) > *(int *)vector_get(v, 2))
        swap(vector_get(v, 1), vector_get(v, 2));

    if (*(int *)vector_get(v, 0) > *(int *)vector_get(v, 1))
        swap(vector_get(v, 0), vector_get(v, 1));
}

Rect boundingRect(vector* v/* Point */)
{
    Mat m = getMatfromVector(v);
    return depth(m) <= 0 ? maskBoundingRect(&m) : pointSetBoundingRect(m);
}

int norm(int x, int y)
{
    return sqrt(x*x + y*y);
}

bool vector_contains(vector* v, Point a) //Checks specifically for the struct Point
{
    for(int i = 0; i < vector_size(v); i++)
    {
        Point pt = *(Point*)vector_get(v, i);
        if(pt.x == a.x && pt.y == a.y)
            return true;
    }
    return false;
}

int sort_couples(const void* i, const void* j) 
{
    return (*(Vec3i*)i).val[0] - (*(Vec3i*)j).val[0]; 
}

void sort(vector* v /* Vec3i */)
{
    int len = vector_size(v);
    Vec3i arr[len];

    for(int i = 0; i < len; i++)
        arr[i] = *(Vec3i*)vector_get(v, i);

    vector_init(v);
    qsort(arr, len, sizeof(Vec3i), sort_couples);

    for(int i = 0; i < len; i++)
        vector_add(v, &arr[i]);
}

int floor(double value)
{
    int i = (int)value;
    return i - (i > value);
}

int round(double value)
{
    /* it's ok if round does not comply with IEEE754 standard;
       the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float* a0, float* a1)
{
    *a1 = (float)(p2.y - p1.y) / (p2.x - p1.x);
    *a0 = *a1 * -1 * p1.x + p1.y;
}


// Fit a line_estimate to a group of 3 regions
// out triplet->estimates is updated with the new line estimates
bool fitLineEstimates(vector** regions/* ERStat */, region_triplet* triplet)
{
    vector* char_boxes = malloc(sizeof(vector));
    vector_init(char_boxes);
    vector_add(char_boxes, &((*ERStat*)vector_get(regions[triplet->a[0]], triplet->a[1])).rect);
    vector_add(char_boxes, &((*ERStat*)vector_get(regions[triplet->b[0]], triplet->b[1])).rect);
    vector_add(char_boxes, &((*ERStat*)vector_get(regions[triplet->c[0]], triplet->c[1])).rect);
    
    triplet->estimates.x_min = min(min((Rect*)vector_get(char_boxes, 0)->x,(Rect*)vector_get(char_boxes, 1)->x), (Rect*)vector_get(char_boxes, 2)->x);
    triplet->estimates.x_max = max(max((Rect*)vector_get(char_boxes, 0)->x + (Rect*)vector_get(char_boxes, 0)->width, (Rect*)vector_get(char_boxes, 1)->x + (Rect*)vector_get(char_boxes, 1)->width), (Rect*)vector_get(char_boxes, 2)->x + (Rect*)vector_get(char_boxes, 2)->width);
    triplet->estimates.h_max = max(max((Rect*)vector_get(char_boxes, 0)->height, (Rect*)vector_get(char_boxes, 0)->height), (Rect*)vector_get(char_boxes, 0)->x);

    // Fit one bottom line
    float err = fitLineLMS(init_Point((Rect*)vector_get(char_boxes, 0)->x + (Rect*)vector_get(char_boxes, 0)->width, (Rect*)vector_get(char_boxes, 0)->y + (Rect*)vector_get(char_boxes, 0)->height),
                           init_Point((Rect*)vector_get(char_boxes, 1)->x + (Rect*)vector_get(char_boxes, 1)->width, (Rect*)vector_get(char_boxes, 1)->y + (Rect*)vector_get(char_boxes, 1)->height),
                           init_Point((Rect*)vector_get(char_boxes, 2)->x + (Rect*)vector_get(char_boxes, 2)->width, (Rect*)vector_get(char_boxes, 2)->y + (Rect*)vector_get(char_boxes, 2)->height), 
                           &(triplet->estimates.bottom1_a0), &(triplet->estimates.bottom1_a1));

    if ((triplet->estimates.bottom1_a0 == -1) && (triplet->estimates.bottom1_a1 == 0))
        return false;

    // Slope for all lines must be the same
    triplet->estimates.bottom2_a1 = triplet->estimates.bottom1_a1;
    triplet->estimates.top1_a1    = triplet->estimates.bottom1_a1;
    triplet->estimates.top2_a1    = triplet->estimates.bottom1_a1;

    if (abs(err) > (float)triplet->estimates.h_max/6)
    {
        // We need two different bottom lines
        triplet->estimates.bottom2_a0 = triplet->estimates.bottom1_a0 + err;
    }
    else
    {
        // Second bottom line is the same
        triplet->estimates.bottom2_a0 = triplet->estimates.bottom1_a0;
    }

    // Fit one top line within the two (Y)-closer coordinates
    int d_12 = abs((Rect*)vector_get(char_boxes, 0)->y - (Rect*)vector_get(char_boxes, 1)->y);
    int d_13 = abs((Rect*)vector_get(char_boxes, 0)->y - (Rect*)vector_get(char_boxes, 2)->y);
    int d_23 = abs((Rect*)vector_get(char_boxes, 1)->y - (Rect*)vector_get(char_boxes, 2)->y);
    if((d_12 < d_13) && (d_12 < d_23))
    {
        Point p = init_Point(((Rect*)vector_get(char_boxes, 0)->x + (Rect*)vector_get(char_boxes, 1)->x)/2,
                        ((Rect*)vector_get(char_boxes, 0)->y + (Rect*)vector_get(char_boxes, 1)->y)/2);

        triplet->estimates.top1_a0 = triplet->estimates.bottom1_a0 +
                (p.y - (triplet->estimates.bottom1_a0+p.x*triplet->estimates.bottom1_a1));

        p = init_Point((Rect*)vector_get(char_boxes, 2)->x, (Rect*)vector_get(char_boxes, 2)->y);
        err = (p.y - (triplet->estimates.top1_a0+p.x*triplet->estimates.top1_a1));
    }
    else if(d_13 < d_23)
    {
        Point p = init_Point(((Rect*)vector_get(char_boxes, 0)->x + (Rect*)vector_get(char_boxes, 2)->x)/2,
                        ((Rect*)vector_get(char_boxes, 0)->y + (Rect*)vector_get(char_boxes, 2)->y)/2);
        triplet->estimates.top1_a0 = triplet->estimates.bottom1_a0 +
                (p.y - (triplet->estimates.bottom1_a0+p.x*triplet->estimates.bottom1_a1));

        p = init_Point((Rect*)vector_get(char_boxes, 1)->x, (Rect*)vector_get(char_boxes, 1)->y);
        err = (p.y - (triplet->estimates.top1_a0+p.x*triplet->estimates.top1_a1));
    }
    else
    {
        Point p = init_Point(((Rect*)vector_get(char_boxes, 1)->x + (Rect*)vector_get(char_boxes, 2)->x)/2,
                        ((Rect*)vector_get(char_boxes, 1)->y + (Rect*)vector_get(char_boxes, 2)->y)/2);

        triplet->estimates.top1_a0 = triplet->estimates.bottom1_a0 +
                (p.y - (triplet->estimates.bottom1_a0+p.x*triplet->estimates.bottom1_a1));

        p = init_Point((Rect*)vector_get(char_boxes, 0)->x, (Rect*)vector_get(char_boxes, 0)->y);
        err = (p.y - (triplet->estimates.top1_a0+p.x*triplet->estimates.top1_a1));
    }

    if (abs(err) > (float)triplet->estimates.h_max/6)
    {
        // We need two different top lines
        triplet->estimates.top2_a0 = triplet->estimates.top1_a0 + err;
    }
    else
    {
        // Second top line is the same
        triplet->estimates.top2_a0 = triplet->estimates.top1_a0;
    }

    return true;
}
