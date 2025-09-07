## Urađeno do sada (Aleksandar)
- Postavljen ceo projekat od nule:
  - CMake konfiguracija
  - Git repozitorijum
  - Povezivanje biblioteka
  - Struktura `src/` i `shaders/`
- Sistem za učitavanje šejdera iz fajlova (`chess.vert`, `chess.frag`) sa uniformama iz C++ koda
- Proceduralna geometrija šahovske table sa debljinom (mesh generisan u kodu sa normalama i UV koordinatama)
- Proceduralna šahovska tekstura u shaderu (checkerboard, broj polja, osvetljenje i posebna boja za stranice)
- Particle system:
  - Generisanje čestica sa random brzinom, uglom i životnim vekom
  - Gravitacija i sudaranje sa gornjom pločom table
  - Gašenje čestica nakon isteka života
- GPU deo particle sistema (VS + FS za čestice: veličina, fade-out, boja)
- Tastatura:
  - Uključivanje/isključivanje table i čestica (1, 2, 3)
  - Start/stop emitovanja (P/O)
  - Reset (R)
  - Izlaz (ESC)
- Kamera: podešena perspektivna projekcija, pozicija, target, up vektor, FOV

---

## Ispunjeno iz specifikacije
- **Generalna 3D grafika** - tabla/cestice
  ### Tehnike:
- **Proceduralne teksture** - šahovski uzorak u fragment shaderu  
- **Animated particle systems** - čestice sa kretanjem i interakcijom  

 Jos jedna treba da se uradi

---

## Šta još treba da se uradi

### Nemanja
- Implementirati **post-processing pipeline** (framebuffer, quad, učitavanje post-processing šejdera)  
- Dodati makar jedan post-processing efekat (npr. blur, invert boja, grayscale)  
- Dokumentovati pipeline u README  

### Jovan
- Dodati još jednu zahtevanu tehniku iz specifikacije (primeri: SSAO, spotlight senke, voda ili teren)  
- Implementirati **hijerarhiju objekata** (npr. rotacija figure, animacija dela scene)  
- Dokumentovati svoj deo u README  

---

## Dokumentacija
- Svako treba da napiše u README koje tehnike je implementirao i ukratko objasni svoj pristup  
- Po potrebi dopuniti komentarima u kodu (posebno shaderi) 
