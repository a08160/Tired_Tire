// import 'dart:convert';
// import 'dart:math';
// import 'package:csv/csv.dart';
// import 'package:flutter/services.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter/foundation.dart'
//     show defaultTargetPlatform, TargetPlatform;

// class ServiceCenterPage extends StatefulWidget {
//   const ServiceCenterPage({Key? key}) : super(key: key);

//   @override
//   _ServiceCenterPageState createState() => _ServiceCenterPageState();
// }

// class _ServiceCenterPageState extends State<ServiceCenterPage> {
//   static const platform = MethodChannel(
//     'com.example.flutter_application_1/kakao_map',
//   );

//   // 이후 위치를 native에 전달하는 함수
//   Future<void> sendUserLocationToNative(
//     double lat,
//     double lng,
//     List<ServiceCenter> nearbyCenters,
//   ) async {
//     try {
//       await platform.invokeMethod('showMap', {
//         'latitude': lat,
//         'longitude': lng,
//         'centers':
//             nearbyCenters
//                 .map(
//                   (center) => {
//                     'name': center.name,
//                     'lat': center.lat,
//                     'lng': center.lng,
//                     'address': center.address,
//                   },
//                 )
//                 .toList(),
//       });
//     } on PlatformException catch (e) {
//       print("Failed to invoke native method: '${e.message}'.");
//     }
//   }

//   List<ServiceCenter> centers = [];
//   List<ServiceCenter> nearbyCenters = [];
//   double? userLat;
//   double? userLng;

//   @override
//   void initState() {
//     super.initState();
//     _loadCsv();
//   }

//   Future<void> _loadCsv() async {
//     final rawData = await rootBundle.loadString('assets/service_center.csv');
//     final rows = const CsvToListConverter(eol: '\n').convert(rawData);

//     List<ServiceCenter> parsedCenters = [];
//     for (int i = 1; i < rows.length; i++) {
//       final row = rows[i];
//       double? lat = _toDouble(row[4]);
//       double? lng = _toDouble(row[5]);
//       if (lat != null && lng != null) {
//         parsedCenters.add(
//           ServiceCenter(
//             name: row[0].toString(),
//             lat: lat,
//             lng: lng,
//             address: row[2].toString(),
//             phone: row[14].toString(),
//           ),
//         );
//       }
//     }

//     setState(() {
//       centers = parsedCenters;
//     });
//   }

//   double? _toDouble(dynamic value) {
//     if (value is double) return value;
//     if (value is int) return value.toDouble();
//     if (value is String) return double.tryParse(value);
//     return null;
//   }

//   double calculateDistance(lat1, lon1, lat2, lon2) {
//     const p = 0.017453292519943295;
//     final a =
//         0.5 -
//         cos((lat2 - lat1) * p) / 2 +
//         cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2;
//     return 12742 * asin(sqrt(a));
//   }

//   void _onUserLocation(double lat, double lng) {
//     setState(() {
//       userLat = lat;
//       userLng = lng;
//     });

//     final nearby =
//         centers.map((center) {
//             final dist = calculateDistance(lat, lng, center.lat, center.lng);
//             return MapEntry(center, dist);
//           }).toList()
//           ..sort((a, b) => a.value.compareTo(b.value));

//     final top5 = nearby.take(5).map((e) => e.key).toList();

//     setState(() {
//       nearbyCenters = top5;
//     });

//     // ✅ 네이티브로 넘기기
//     sendUserLocationToNative(lat, lng, top5);
//   }

//   @override
//   Widget build(BuildContext context) {
//     if (defaultTargetPlatform == TargetPlatform.android) {
//       // ✅ 안드로이드에서는 네이티브 View 호출
//       return Scaffold(
//         appBar: AppBar(title: const Text('정비소 찾기')),
//         backgroundColor: Colors.white,
//         body: Column(
//           children: [
//             const SizedBox(
//               height: 300,
//               child: AndroidView(viewType: 'kakao-map-view'),
//             ),
//             const SizedBox(height: 30),
//             Expanded(child: _buildNearbyList()),
//           ],
//         ),
//       );
//     }
//     // ✅ 안드로이드 외 플랫폼 대비 기본 fallback (혹시 나중에 다른 플랫폼 디버깅용)
//     return Scaffold(
//       appBar: AppBar(title: const Text('정비소 찾기')),
//       backgroundColor: Colors.white,
//       body: const Center(child: Text('안드로이드 기기에서만 지원됩니다.')),
//     );
//   }

//   Widget _buildNearbyList() {
//     return ListView.builder(
//       shrinkWrap: true,
//       physics: const NeverScrollableScrollPhysics(),
//       itemCount: nearbyCenters.length,
//       itemBuilder: (context, index) {
//         final center = nearbyCenters[index];
//         return Card(
//           shape: RoundedRectangleBorder(borderRadius: BorderRadius.zero),
//           elevation: 0,
//           color: Colors.white,
//           margin: const EdgeInsets.symmetric(vertical: 8),
//           child: ListTile(
//             leading: CircleAvatar(
//               backgroundColor: Colors.grey.shade300,
//               child: Text(
//                 '${index + 1}',
//                 style: const TextStyle(color: Colors.white),
//               ),
//             ),
//             title: Text(
//               center.name,
//               style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
//             ),
//             subtitle: Text(center.address),
//             isThreeLine: false,
//           ),
//         );
//       },
//     );
//   }
// }

// class ServiceCenter {
//   final String name;
//   final double lat;
//   final double lng;
//   final String address;
//   final String phone;

//   ServiceCenter({
//     required this.name,
//     required this.lat,
//     required this.lng,
//     required this.address,
//     required this.phone,
//   });
// }
