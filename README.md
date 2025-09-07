# RacunarskaGrafika

Urađeno do sada (Aleksandar)

Postavljen ceo projekat od nule (CMake, git, povezivanje biblioteka, struktura src/ i shaders/)

Sistem za učitavanje šejdera iz fajlova (chess.vert, chess.frag), sa uniformama iz C++ koda

Proceduralna geometrija šahovske table sa debljinom (mesh generisan u kodu sa normalama i UV koordinatama)

Proceduralna šahovska tekstura u shaderu (checkerboard, broj polja, osvetljenje i posebna boja za stranice)

Particle system:

generisanje čestica sa random brzinom, uglom i životnim vekom

gravitacija i sudaranje sa gornjom pločom table

gašenje čestica nakon isteka života

GPU deo particle sistema (VS + FS za čestice: veličina, fade-out, boja)

Tastatura: kontrola uključivanja/isključivanja table i čestica (1,2,3), start/stop emitovanja (P/O), reset (R), izlaz (ESC)

Kamera: podešena perspektivna projekcija, pozicija, target, up vektor, FOV

Šta je ispunjeno od specifikacije

Proceduralno generisana geometrija → šahovska tabla

Proceduralne teksture → šahovski uzorak u fragment shaderu

Animated particle systems → čestice sa kretanjem i interakcijom
To su tri tehnike/proširenja koja su obavezna po projektu.

Šta još treba da se uradi
Nemanja

Implementirati post-processing pipeline (framebuffer, quad, učitavanje post-processing šejdera)

Dodati makar jedan post-processing efekat (npr. blur, invert boja, grayscale)

Dokumentovati pipeline u README

Jovan

Dodati još jednu zahtevanu tehniku iz specifikacije (primeri: SSAO, spotlight senke, voda ili teren)

Implementirati hijerarhiju objekata (npr. rotacija figure, animacija dela scene)

Dokumentovati svoj deo u README

Dokumentacija

Svako treba da napiše u README koje tehnike je implementirao i ukratko objasni pristup

Po potrebi dopuniti komentarima u kodu (posebno shaderi)
