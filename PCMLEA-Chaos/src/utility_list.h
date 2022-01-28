/*
  list.h

  Author:
  Luis Miguel Antonio <lmiguel@computacion.cs.cinvestav.mx>

  Copyright (c) 2013 Luis Miguel Antonio		*/

# ifndef _LIST_H_
# define _LIST_H_

typedef struct node {
	int index;
	struct node* parent;
	struct node* child;
} list, node;

void insert(node* n, int x);
node* deleteNode(node* n);
node* deleteInd(list* l, int index);
void deleteList(list* l);
list* createList(int index);

# endif
