title: Adventures in procedural (hash) texturing
keywords: graphics,maths,webgl

<script src="./hash_textures.js"></script>
<link rel="stylesheet" href="./hash_textures.css">
<style>
.hv-equation {
  font-size: small;
}
.hv {
  max-width: 25em;
  margin-top: -0.5em;
  margin-bottom: 1.2em;
  padding-bottom: 0.8em;
}
.hv-large {
  max-width: 100%;
}
.favourites {
    display: inline-block;
    width: 100%;
}
.favourites > div {
    display: inline-block;
}
.favourites .hv {
    width: 20em;
}
.favourites .hv-equation {
  font-size: 0.65em;
}
</style>

# Adventures in procedural (hash) texturing

<a class="hv-title" href="./hv.html">Hash Playground</a>

In need of textures for my game and lacking any artistic skill, I turned to maths to help me. The result was an elegant 2D hash function that I just **need** to talk about, humour me...

**TL;DR: The boolean predicate, $(c_x \, x + c_y \, y + c_{xy} \, x \, y + c_{x^2} \, x^2 + c_{y^2} \, y^2) \,\mathrm{mod} \, m < \tau \, m$, is wonderously varied and beautiful. For example, varying $c_{xy}$ with the other parameters fixed:**

<div class="hv hv-show-equation hv-large"
  data-hv-init="x1 y1 xy* xx1 yy1 p64 t0.5 h256w256 k000000 j0ed3e1 r2"
></div>

For the rest of this post, we'll try to unpick the function a little. If you'd prefer to play with it yourself, check out the [hash playground](hv.html).

## The idea

My goal was to make a game that obeyed a 2-bit colour palette. Being strict about it, this means no interpolation or antialiasing — the pixels on screen should only be one of 4 colours. So I needed textures that could align perfectly to the screen.

_What if you could just "hash" the (x, y) screen-space pixel coordinate, and use it to choose a colour?_

It turns out that this works fine, as long as the camera and viewports are fixed, and that one of the simplest possible hash functions is remarkably complex and beautiful:

$$(c_x \, x + c_y \, y + c_{xy} \, x \, y + c_{x^2} \, x^2 + c_{y^2} \, y^2) \,\mathrm{mod} \, m < \tau \, m$$

In Python, for example:

```python
c_x, c_y, c_xy, c_xx, c_yy = 1, 1, 0, 1, 1
m, t, w, h = 64, 0.5, 128, 128

x = np.arange(w)[:, None]
y = np.arange(h)[None, :]

h = (c_x*x + c_y*y + c_xy*x*y + c_xx*x**2 + c_yy*y**2) % m < t*m

display(PIL.Image.fromarray(h))
```

## A little understanding

Let's try to build up the maths to work out what's going on. We'll call the expression between $()$ the body. Note that everything in this section uses a canvas width $128$ and divisor $m\\!=\\!64$. So it does:

**What if our body is just $(x)$?** The hash won't depend on $y$ and should be $\mathrm{true}$ for $x \\!<\\! 32$ and $\mathrm{false}$ for $32 \\!\leq\\! x \\!<\\! 64$. It should repeat twice over our canvas width of $128$.

<div class="hv hv-show-equation"
  data-hv-init="x1 y0 xy0 xx0 yy0 p64 t0.5 h8w128 k000000 j0ed3e1"
></div>

**How about $(x^2)$?** I expected to see a twice-repeating pattern again, but I was wrong — we get $4$ copies:

<div class="hv hv-show-equation"
  data-hv-init="x0 y0 xy0 xx1 yy0 p64 t0.5 h8w128 k000000 j0ed3e1"
></div>

I.e. the pattern seems to repeat after $m/2$ not $m$. Let's see what happens if we add $m/2$ to $x$:

$$
\begin{align}
&(x + m/2)^2 \\,\mathrm{mod}\\, m  \\\\
&= (x^2 + m\,x + m^2/4) \\,\mathrm{mod}\\, m \\\\
&= x^2 \\,\mathrm{mod}\\, m \\;(\textrm{if } m \textrm{ is a multiple of } 4) \\\\
\end{align}
$$

Since the $+\, m/2$ disappears, the pattern must repeat every $m/2$ pixels rather than $m$.

**Going 2D, $(x^2 + y^2)$.** To get some interesting patterns, we need to use $x$ and $y$. Here's one of the simplest hash functions. Since the equation of a circle is $x^2 + y^2 = \textit{const}$, circles are _iso-hash_ lines which are clearly visible. Since the expression is built using $x^2$ and $y^2$, we get $2$ copies per $m$ in each axis, so $16$ sets of circles in total.

<div class="hv hv-show-equation"
  data-hv-init="x0 y0 xy0 xx1 yy1 p64 t0.5 h128w128 k000000 j0ed3e1"
></div>

**Increasing the frequency, $(2 x^2 + 2 y^2)$.** Unsurprisingly, doubling $c_{x^2}$ and $c_{y^2}$ doubles the frequency of the pattern, equivalent to halving $m$ from $64$ to $32$.

<div class="hv hv-show-equation"
  data-hv-init="x0 y0 xy0 xx2 yy2 p64 t0.5 h128w128 k000000 j0ed3e1"
></div>

**Cross terms, $(x\,y)$.** Some of my favourite patterns come from fiddling with $c_{xy}$. Even the simple body $x\,y$ is interesting. I don't have much of an explanation, other than the observation that the 'lines' look much like a family a reciprocal curves $y\propto1/x$, which makes sense considering the iso-hash line, $x\,y = \textit{const}$.

<div class="hv hv-show-equation"
  data-hv-init="x0 y0 xy1 xx0 yy0 p64 t0.5 h128w128 k000000 j0ed3e1"
></div>

That's all I've got for this post, but I'm sure there's much more that could be said from the maths of modular arithmetic.

## Some favourites

Here are some of my favourite patterns:

<div class="favourites">
<div><div class="hv hv-show-equation"
data-hv-init="x3 y3 xy32 xx6 yy6 p32 t0.5 h128w128 kffffff jff0000"
></div></div>
<div><div class="hv hv-show-equation"
data-hv-init="x3 y3 xy32 xx7 yy7 p32 t0.5 h128w128 kffffff j007529"
></div></div>

<div><div class="hv hv-show-equation"
data-hv-init="x0 y0 xy3 xx1 yy1 p32 t0.5 h128w128 ke8f359 jff1a3c"
></div></div>
<div><div class="hv hv-show-equation"
data-hv-init="x3 y3 xy17 xx2 yy2 p64 t0.8 h128w128 kffffff jff751a"
></div></div>
</div>

And it's particularly fun to sweep the threshold $\tau$, which can be useful for effects like explosions.

<div class="hv hv-show-equation hv-large"
  data-hv-init="x0 y0 xy31 xx7 yy7 p64 t* h192w192 kffffff j000000 r10.000"
></div>

## Wrap up

I hope you enjoyed staring at the complexities of a simple function for a few minutes! If you haven't already, try the [hash playground](hv.html). Maybe you can generate a new texture for your game or print a case for your mobile phone (my next step). And if you consider yourself part of the extremely limited target audience, perhaps check out my asm programming game-jam game, [C-crits](https://douglasorr.itch.io/c-crits), which uses these hashes for terrain and effects.

Happy Hashing!
