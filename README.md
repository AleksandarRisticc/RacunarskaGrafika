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

## Urađeno do sada (Jovan)
- Implementirani novi objekti i njihove funkcionalnosti:
  - Beli i crni top na početku na suprotnim krajevima table
  - Hijerarhija elemenata i rotirajući elementi na vrhu obe figure
  - Senke pokretnih elemenata
  - Kretanje topova pomoću tastature
- Dodata celokupna logika igre:
  - Simple GUI pred pocetak runde
  - Tajmer
  - Ispis pobednika i povratak na meni
  - Okidanje particle eksplozije kada beli top pojede crnog (završe na istom polju table)
- Tastatura:
  - Kretanje belog topa (strelice)
  - Kretanje crnog topa (WASD)
- Iskorišćena dodatna tekstura
  - Uveden je roughness teksturni uzorak (proceduralno generisan u CPU-u), koji u šejderu kontroliše specular (Blinn-Phong) i shininess
- Proširen set aktivnih šejdera (dodati `mesh.vert`, `mesh.frag`, `shadow_depth.vert`, `shadow_depth.frag`)
- Implementirana viseća lampa kao centralni izvor svetlosti

---

## Ispunjeno iz specifikacije
- **Generalna 3D grafika** - tabla/cestice
  ### Tehnike:
- **Proceduralne teksture** - šahovski uzorak u fragment shaderu  
- **Animated particle systems** - čestice sa kretanjem i interakcijom
- **Spotlight shadows and point light shadows** - usmereno centralno svetlo iznad table + senke
  ### Ostalo:
- **Interaktivnost** - kretanje šahovskih figura, dugmići, logika igre, tajmer, GUI...
- **Hijerarhije objekata** - rotirajući elementi vezani za figure
- **Korišćenje drugih tekstura pored albedo** - roughness teksturni uzorak za dodatni shine

---

## Šta još treba da se uradi

### Nemanja
- Implementirati **post-processing pipeline** (framebuffer, quad, učitavanje post-processing šejdera)  
- Dodati makar jedan post-processing efekat (npr. blur, invert boja, grayscale)  
- Dokumentovati pipeline u README  

---

## Dokumentacija
- Svako treba da napiše u README koje tehnike je implementirao i ukratko objasni svoj pristup  
- Po potrebi dopuniti komentarima u kodu (posebno shaderi) 
