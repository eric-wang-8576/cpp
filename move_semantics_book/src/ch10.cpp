/*
 *
 */

/*
 * A universal reference is the only way we can bind a reference to objects of any value category and
 *   still preserve whether or not it is const - the only other reference that binds to all objects,
 *   const&, loses the information about whether the passed argument is const or not
 */

// PG 154 
